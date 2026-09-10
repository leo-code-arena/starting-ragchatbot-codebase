"""Shared fixtures for the backend test suite.

Every test here is mock-only: no test touches the Anthropic API or a real ChromaDB,
so the suite runs green regardless of whether ANTHROPIC_API_KEY is set.
"""

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from vector_store import SearchResults


# --- Anthropic API response builders -------------------------------------------------
# The real SDK returns pydantic models; the production code only touches
# .stop_reason, .content, and per-block .type/.text/.name/.id/.input, so light
# stand-ins are enough and keep the tests readable.


def text_block(text):
    """A single `text` content block."""
    return SimpleNamespace(type="text", text=text)


def tool_use_block(name, tool_input, block_id="toolu_test_1"):
    """A single `tool_use` content block."""
    return SimpleNamespace(type="tool_use", name=name, input=tool_input, id=block_id)


def text_response(text="A plain answer."):
    """A completed response carrying one text block."""
    return SimpleNamespace(stop_reason="end_turn", content=[text_block(text)])


def tool_use_response(name="search_course_content", tool_input=None, block_id="toolu_test_1"):
    """A response asking for one tool call."""
    return SimpleNamespace(
        stop_reason="tool_use",
        content=[tool_use_block(name, tool_input or {"query": "test"}, block_id)],
    )


@pytest.fixture
def mock_anthropic_client():
    """A stand-in Anthropic client whose `messages.create` is a MagicMock.

    Set `client.messages.create.return_value` (single call) or `.side_effect`
    (a list, for the two-call tool round trip).
    """
    client = MagicMock()
    client.messages.create.return_value = text_response()
    return client


@pytest.fixture
def ai_generator(mock_anthropic_client, monkeypatch):
    """An AIGenerator wired to the mock client instead of the real SDK."""
    import ai_generator as ai_generator_module

    monkeypatch.setattr(
        ai_generator_module.anthropic, "Anthropic", lambda **kwargs: mock_anthropic_client
    )
    return ai_generator_module.AIGenerator(api_key="test-key", model="test-model")


# --- Vector store stand-ins ----------------------------------------------------------


SAMPLE_DOCUMENTS = [
    "MCP is an open protocol that standardizes how applications provide context to LLMs.",
    "An MCP server exposes tools, resources, and prompts to a client.",
]

SAMPLE_METADATA = [
    {"course_title": "MCP: Build Rich-Context AI Apps", "lesson_number": 1, "chunk_index": 0},
    {"course_title": "MCP: Build Rich-Context AI Apps", "lesson_number": 2, "chunk_index": 1},
]


def make_results(documents=None, metadata=None, error=None):
    """Build a real SearchResults so formatting logic is exercised for real."""
    if error is not None:
        return SearchResults.empty(error)
    documents = SAMPLE_DOCUMENTS if documents is None else documents
    metadata = SAMPLE_METADATA if metadata is None else metadata
    return SearchResults(
        documents=documents,
        metadata=metadata,
        distances=[0.1] * len(documents),
    )


@pytest.fixture
def mock_vector_store():
    """A VectorStore stand-in returning canned search results and links."""
    store = MagicMock()
    store.search.return_value = make_results()
    store.get_lesson_link.return_value = "https://example.com/lesson"
    store.get_course_link.return_value = "https://example.com/course"
    store._resolve_course_name.return_value = "MCP: Build Rich-Context AI Apps"
    store.get_course_metadata.return_value = {
        "title": "MCP: Build Rich-Context AI Apps",
        "instructor": "Elie Schoppik",
        "course_link": "https://example.com/course",
        "lessons": [
            {"lesson_number": 1, "lesson_title": "Why MCP", "lesson_link": "https://example.com/l1"},
            {"lesson_number": 2, "lesson_title": "MCP Servers", "lesson_link": "https://example.com/l2"},
        ],
    }
    return store
