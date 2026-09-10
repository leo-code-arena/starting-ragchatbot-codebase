"""Tests for the FastAPI layer: does a content query come back as a usable
response, and does a failure tell the caller what actually went wrong?"""

import os
from pathlib import Path
from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient

BACKEND_DIR = Path(__file__).resolve().parent.parent


@pytest.fixture
def app_module(monkeypatch):
    """Import app.py with a stubbed RAGSystem.

    app.py builds its RAGSystem at import time and resolves `../frontend` and
    `../docs` relative to the cwd, so the import happens from backend/.
    """
    import rag_system as rag_system_module

    fake_rag = MagicMock()
    fake_rag.session_manager.create_session.return_value = "session_1"
    fake_rag.query.return_value = (
        "MCP is an open protocol.",
        [{"text": "MCP Course - Lesson 1", "link": "https://example.com/l1"}],
    )
    fake_rag.get_course_analytics.return_value = {
        "total_courses": 2,
        "course_titles": ["MCP Course", "Chroma Course"],
    }
    fake_rag.add_course_folder.return_value = (0, 0)

    monkeypatch.setattr(rag_system_module, "RAGSystem", lambda config: fake_rag)
    monkeypatch.chdir(BACKEND_DIR)

    # Force a fresh import so the patched RAGSystem is the one app.py picks up.
    import sys

    sys.modules.pop("app", None)
    import app as app_module

    app_module.fake_rag = fake_rag
    yield app_module
    sys.modules.pop("app", None)


@pytest.fixture
def client(app_module):
    with TestClient(app_module.app) as test_client:
        yield test_client


# --- Happy path ----------------------------------------------------------------------


def test_query_returns_answer_sources_and_session(client, app_module):
    response = client.post("/api/query", json={"query": "What is MCP?"})

    assert response.status_code == 200
    body = response.json()
    assert body["answer"] == "MCP is an open protocol."
    assert body["sources"] == [
        {"text": "MCP Course - Lesson 1", "link": "https://example.com/l1"}
    ]
    assert body["session_id"] == "session_1"


def test_query_reuses_a_supplied_session_id(client, app_module):
    response = client.post(
        "/api/query", json={"query": "Tell me more.", "session_id": "session_42"}
    )

    assert response.json()["session_id"] == "session_42"
    app_module.fake_rag.query.assert_called_with("Tell me more.", "session_42")


def test_sources_may_omit_the_link(client, app_module):
    app_module.fake_rag.query.return_value = ("Answer.", [{"text": "Some Course", "link": None}])

    response = client.post("/api/query", json={"query": "What is MCP?"})

    assert response.status_code == 200
    assert response.json()["sources"] == [{"text": "Some Course", "link": None}]


def test_courses_endpoint_reports_the_catalog(client):
    response = client.get("/api/courses")

    assert response.status_code == 200
    assert response.json() == {
        "total_courses": 2,
        "course_titles": ["MCP Course", "Chroma Course"],
    }


# --- Failure path --------------------------------------------------------------------


def test_query_failure_returns_the_underlying_reason(client, app_module):
    """The UI shows a bare "Query failed" because it never reads this; the API
    must at minimum put the real cause in `detail`."""
    app_module.fake_rag.query.side_effect = RuntimeError(
        "ANTHROPIC_API_KEY is invalid or expired"
    )

    response = client.post("/api/query", json={"query": "What is MCP?"})

    assert response.status_code == 500
    assert "ANTHROPIC_API_KEY" in response.json()["detail"]


def test_query_requires_a_query_field(client):
    response = client.post("/api/query", json={})

    assert response.status_code == 422
