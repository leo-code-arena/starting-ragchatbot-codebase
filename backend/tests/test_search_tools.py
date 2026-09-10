"""Tests for CourseSearchTool.execute (and the ToolManager that dispatches to it)."""

import pytest

from search_tools import CourseOutlineTool, CourseSearchTool, ToolManager

from .conftest import make_results


@pytest.fixture
def search_tool(mock_vector_store):
    return CourseSearchTool(mock_vector_store)


# --- Happy path ----------------------------------------------------------------------


def test_execute_formats_results_with_course_and_lesson_headers(search_tool):
    output = search_tool.execute("What is MCP?")

    assert "[MCP: Build Rich-Context AI Apps - Lesson 1]" in output
    assert "[MCP: Build Rich-Context AI Apps - Lesson 2]" in output
    assert "MCP is an open protocol" in output
    # Chunks are separated by a blank line
    assert "\n\n" in output


def test_execute_omits_lesson_from_header_when_metadata_has_none(search_tool, mock_vector_store):
    mock_vector_store.search.return_value = make_results(
        documents=["Course-level overview text."],
        metadata=[{"course_title": "Some Course", "chunk_index": 0}],
    )

    output = search_tool.execute("overview")

    assert output.startswith("[Some Course]")
    assert "Lesson" not in output


def test_execute_forwards_filters_to_the_vector_store(search_tool, mock_vector_store):
    search_tool.execute("servers", course_name="MCP", lesson_number=2)

    mock_vector_store.search.assert_called_once_with(
        query="servers", course_name="MCP", lesson_number=2
    )


# --- Source tracking -----------------------------------------------------------------


def test_execute_records_sources_with_lesson_links(search_tool, mock_vector_store):
    search_tool.execute("What is MCP?")

    assert search_tool.last_sources == [
        {"text": "MCP: Build Rich-Context AI Apps - Lesson 1", "link": "https://example.com/lesson"},
        {"text": "MCP: Build Rich-Context AI Apps - Lesson 2", "link": "https://example.com/lesson"},
    ]
    mock_vector_store.get_lesson_link.assert_called()
    mock_vector_store.get_course_link.assert_not_called()


def test_execute_falls_back_to_course_link_when_no_lesson_number(search_tool, mock_vector_store):
    mock_vector_store.search.return_value = make_results(
        documents=["Course-level overview text."],
        metadata=[{"course_title": "Some Course", "chunk_index": 0}],
    )

    search_tool.execute("overview")

    assert search_tool.last_sources == [
        {"text": "Some Course", "link": "https://example.com/course"}
    ]
    mock_vector_store.get_course_link.assert_called_once_with("Some Course")


# --- Error and empty handling --------------------------------------------------------


def test_execute_returns_store_error_verbatim(search_tool, mock_vector_store):
    mock_vector_store.search.return_value = make_results(error="No course found matching 'Nope'")

    assert search_tool.execute("anything", course_name="Nope") == "No course found matching 'Nope'"


def test_execute_reports_empty_results(search_tool, mock_vector_store):
    mock_vector_store.search.return_value = make_results(documents=[], metadata=[])

    assert search_tool.execute("obscure topic") == "No relevant content found."


def test_execute_names_the_active_filters_when_empty(search_tool, mock_vector_store):
    mock_vector_store.search.return_value = make_results(documents=[], metadata=[])

    output = search_tool.execute("obscure", course_name="MCP", lesson_number=3)

    assert output == "No relevant content found in course 'MCP' in lesson 3."


def test_empty_results_clear_stale_sources(search_tool, mock_vector_store):
    """A follow-up search that finds nothing must not leave the previous search's
    sources behind, or the UI cites documents the answer was not based on."""
    search_tool.execute("What is MCP?")
    assert search_tool.last_sources  # populated by the first search

    mock_vector_store.search.return_value = make_results(documents=[], metadata=[])
    search_tool.execute("obscure topic")

    assert search_tool.last_sources == []


def test_error_results_clear_stale_sources(search_tool, mock_vector_store):
    search_tool.execute("What is MCP?")
    assert search_tool.last_sources

    mock_vector_store.search.return_value = make_results(error="Search error: boom")
    search_tool.execute("anything")

    assert search_tool.last_sources == []


# --- Tool definition -----------------------------------------------------------------


def test_tool_definition_declares_the_expected_schema(search_tool):
    definition = search_tool.get_tool_definition()

    assert definition["name"] == "search_course_content"
    properties = definition["input_schema"]["properties"]
    assert set(properties) == {"query", "course_name", "lesson_number"}
    assert definition["input_schema"]["required"] == ["query"]


# --- ToolManager ---------------------------------------------------------------------


@pytest.fixture
def tool_manager(mock_vector_store):
    manager = ToolManager()
    manager.register_tool(CourseSearchTool(mock_vector_store))
    manager.register_tool(CourseOutlineTool(mock_vector_store))
    return manager


def test_manager_exposes_both_tool_definitions(tool_manager):
    names = {d["name"] for d in tool_manager.get_tool_definitions()}

    assert names == {"search_course_content", "get_course_outline"}


def test_manager_executes_a_registered_tool(tool_manager):
    output = tool_manager.execute_tool("search_course_content", query="What is MCP?")

    assert "MCP is an open protocol" in output


def test_manager_reports_unknown_tools_instead_of_raising(tool_manager):
    assert tool_manager.execute_tool("no_such_tool", query="x") == "Tool 'no_such_tool' not found"


def test_manager_collects_and_resets_sources(tool_manager):
    tool_manager.execute_tool("search_course_content", query="What is MCP?")
    assert tool_manager.get_last_sources()

    tool_manager.reset_sources()
    assert tool_manager.get_last_sources() == []
