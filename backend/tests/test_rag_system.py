"""Tests for how RAGSystem handles content-related queries end to end
(with the vector store and the Anthropic client mocked out)."""

from unittest.mock import MagicMock

import ai_generator as ai_generator_module
import anthropic
import httpx
import pytest

import rag_system as rag_system_module
from config import Config
from rag_system import RAGSystem

from .conftest import make_results, text_response, tool_use_response


@pytest.fixture
def config():
    return MagicMock(
        CHUNK_SIZE=800,
        CHUNK_OVERLAP=100,
        CHROMA_PATH="./unused",
        EMBEDDING_MODEL="unused",
        MAX_RESULTS=5,
        ANTHROPIC_API_KEY="test-key",
        ANTHROPIC_MODEL="test-model",
        MAX_HISTORY=2,
    )


@pytest.fixture
def rag(config, mock_vector_store, mock_anthropic_client, monkeypatch):
    """A RAGSystem with a stubbed VectorStore and a stubbed Anthropic client.

    The real ToolManager, CourseSearchTool and AIGenerator are used, so the
    tool-dispatch path is genuinely exercised.
    """
    monkeypatch.setattr(rag_system_module, "VectorStore", lambda *a, **kw: mock_vector_store)
    monkeypatch.setattr(
        ai_generator_module.anthropic, "Anthropic", lambda **kw: mock_anthropic_client
    )
    return RAGSystem(config)


# --- Tool registration ---------------------------------------------------------------


def test_both_tools_are_registered(rag):
    names = {d["name"] for d in rag.tool_manager.get_tool_definitions()}

    assert names == {"search_course_content", "get_course_outline"}


def test_query_hands_the_tools_to_the_generator(rag, mock_anthropic_client):
    rag.query("What is MCP?")

    params = mock_anthropic_client.messages.create.call_args.kwargs
    assert {t["name"] for t in params["tools"]} == {
        "search_course_content",
        "get_course_outline",
    }


def test_query_frames_the_prompt_as_a_course_question(rag, mock_anthropic_client):
    rag.query("What is MCP?")

    content = mock_anthropic_client.messages.create.call_args.kwargs["messages"][0]["content"]
    assert "What is MCP?" in content


# --- The content-query path ----------------------------------------------------------


def test_content_query_runs_the_search_and_returns_sources(
    rag, mock_anthropic_client, mock_vector_store
):
    mock_anthropic_client.messages.create.side_effect = [
        tool_use_response("search_course_content", {"query": "What is MCP?"}),
        text_response("MCP is an open protocol."),
    ]

    answer, sources = rag.query("What is MCP?")

    mock_vector_store.search.assert_called_once_with(
        query="What is MCP?", course_name=None, lesson_number=None
    )
    assert answer == "MCP is an open protocol."
    assert sources == [
        {"text": "MCP: Build Rich-Context AI Apps - Lesson 1", "link": "https://example.com/lesson"},
        {"text": "MCP: Build Rich-Context AI Apps - Lesson 2", "link": "https://example.com/lesson"},
    ]


def test_general_knowledge_query_returns_no_sources(rag, mock_anthropic_client, mock_vector_store):
    mock_anthropic_client.messages.create.return_value = text_response("Paris.")

    answer, sources = rag.query("What is the capital of France?")

    assert answer == "Paris."
    assert sources == []
    mock_vector_store.search.assert_not_called()


def test_sources_do_not_leak_into_the_next_query(rag, mock_anthropic_client):
    mock_anthropic_client.messages.create.side_effect = [
        tool_use_response(),
        text_response("First answer."),
        text_response("Second answer, no search."),
    ]

    _, first_sources = rag.query("What is MCP?")
    _, second_sources = rag.query("What is the capital of France?")

    assert first_sources
    assert second_sources == []


def test_empty_search_results_reach_the_model_as_a_no_results_message(
    rag, mock_anthropic_client, mock_vector_store
):
    mock_vector_store.search.return_value = make_results(documents=[], metadata=[])
    mock_anthropic_client.messages.create.side_effect = [
        tool_use_response(),
        text_response("I could not find that in the course materials."),
    ]

    answer, sources = rag.query("Something not in the courses")

    tool_result = mock_anthropic_client.messages.create.call_args_list[1].kwargs["messages"][2]
    assert "No relevant content found" in tool_result["content"][0]["content"]
    assert sources == []


# --- Session handling ----------------------------------------------------------------


def test_query_without_session_sends_no_history(rag, mock_anthropic_client):
    rag.query("What is MCP?")

    system = mock_anthropic_client.messages.create.call_args.kwargs["system"]
    assert "Previous conversation:" not in system


def test_query_with_session_reads_history_and_records_the_exchange(rag, mock_anthropic_client):
    mock_anthropic_client.messages.create.return_value = text_response("First answer.")
    session_id = rag.session_manager.create_session()

    rag.query("What is MCP?", session_id=session_id)
    rag.query("Tell me more.", session_id=session_id)

    system = mock_anthropic_client.messages.create.call_args.kwargs["system"]
    assert "Previous conversation:" in system
    assert "What is MCP?" in system
    assert "First answer." in system

    history = rag.session_manager.get_conversation_history(session_id)
    assert "Tell me more." in history


# --- The reported failure ------------------------------------------------------------


def test_invalid_api_key_surfaces_an_actionable_error(rag, mock_anthropic_client):
    """This is the bug behind the UI's generic "Query failed": a placeholder
    ANTHROPIC_API_KEY made every query 401, and nothing said so."""
    mock_anthropic_client.messages.create.side_effect = anthropic.AuthenticationError(
        "invalid x-api-key",
        response=httpx.Response(401, request=httpx.Request("POST", "https://api.anthropic.com")),
        body=None,
    )

    with pytest.raises(RuntimeError, match="ANTHROPIC_API_KEY"):
        rag.query("What is MCP?")


# --- Config validation (the startup guard for the same bug) --------------------------


def test_config_rejects_the_env_example_placeholder():
    problem = Config(ANTHROPIC_API_KEY="your-anthropic-api-key-here").validate()

    assert problem is not None
    assert "placeholder" in problem


def test_config_rejects_a_missing_key():
    assert "not set" in Config(ANTHROPIC_API_KEY="").validate()


def test_config_accepts_a_real_looking_key():
    assert Config(ANTHROPIC_API_KEY="sk-ant-api03-abc123").validate() is None
