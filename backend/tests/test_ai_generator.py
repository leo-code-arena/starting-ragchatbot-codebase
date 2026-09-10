"""Tests for AIGenerator: does it hand Claude the tools, run the requested tool,
and feed the result back for a final answer?"""

from types import SimpleNamespace
from unittest.mock import MagicMock

import anthropic
import httpx
import pytest

from ai_generator import AIGenerator

from .conftest import text_block, text_response, tool_use_block, tool_use_response


@pytest.fixture
def tool_manager():
    manager = MagicMock()
    manager.execute_tool.return_value = "[MCP Course - Lesson 1]\nMCP is an open protocol."
    return manager


@pytest.fixture
def tool_definitions():
    return [{"name": "search_course_content", "description": "...", "input_schema": {}}]


def call_params(mock_client, index=0):
    """Kwargs of the nth messages.create call."""
    return mock_client.messages.create.call_args_list[index].kwargs


# --- Tool wiring ---------------------------------------------------------------------


def test_tools_are_offered_to_claude(ai_generator, mock_anthropic_client, tool_definitions):
    ai_generator.generate_response("What is MCP?", tools=tool_definitions)

    params = call_params(mock_anthropic_client)
    assert params["tools"] == tool_definitions
    assert params["tool_choice"] == {"type": "auto"}


def test_no_tool_params_when_no_tools_given(ai_generator, mock_anthropic_client):
    ai_generator.generate_response("What is the capital of France?")

    params = call_params(mock_anthropic_client)
    assert "tools" not in params
    assert "tool_choice" not in params


def test_direct_answer_skips_the_tool_manager(
    ai_generator, mock_anthropic_client, tool_manager, tool_definitions
):
    mock_anthropic_client.messages.create.return_value = text_response("Paris.")

    result = ai_generator.generate_response(
        "What is the capital of France?", tools=tool_definitions, tool_manager=tool_manager
    )

    assert result == "Paris."
    tool_manager.execute_tool.assert_not_called()
    assert mock_anthropic_client.messages.create.call_count == 1


# --- The tool round trip -------------------------------------------------------------


def test_tool_use_invokes_the_search_tool_with_claudes_arguments(
    ai_generator, mock_anthropic_client, tool_manager, tool_definitions
):
    mock_anthropic_client.messages.create.side_effect = [
        tool_use_response(
            "search_course_content", {"query": "MCP servers", "course_name": "MCP"}
        ),
        text_response("An MCP server exposes tools."),
    ]

    result = ai_generator.generate_response(
        "What are MCP servers?", tools=tool_definitions, tool_manager=tool_manager
    )

    tool_manager.execute_tool.assert_called_once_with(
        "search_course_content", query="MCP servers", course_name="MCP"
    )
    assert result == "An MCP server exposes tools."
    assert mock_anthropic_client.messages.create.call_count == 2


def test_tool_result_is_fed_back_in_the_second_call(
    ai_generator, mock_anthropic_client, tool_manager, tool_definitions
):
    mock_anthropic_client.messages.create.side_effect = [
        tool_use_response(block_id="toolu_abc"),
        text_response("Final answer."),
    ]

    ai_generator.generate_response(
        "What is MCP?", tools=tool_definitions, tool_manager=tool_manager
    )

    messages = call_params(mock_anthropic_client, 1)["messages"]
    assert [m["role"] for m in messages] == ["user", "assistant", "user"]

    tool_result = messages[2]["content"][0]
    assert tool_result["type"] == "tool_result"
    assert tool_result["tool_use_id"] == "toolu_abc"
    assert tool_result["content"] == tool_manager.execute_tool.return_value


def test_second_call_disables_tools_so_the_loop_terminates(
    ai_generator, mock_anthropic_client, tool_manager, tool_definitions
):
    mock_anthropic_client.messages.create.side_effect = [
        tool_use_response(),
        text_response("Final answer."),
    ]

    ai_generator.generate_response(
        "What is MCP?", tools=tool_definitions, tool_manager=tool_manager
    )

    assert "tools" not in call_params(mock_anthropic_client, 1)


def test_tool_use_without_a_tool_manager_does_not_execute_anything(
    ai_generator, mock_anthropic_client, tool_definitions
):
    """Without a manager the code takes the direct-return branch; it must not crash."""
    mock_anthropic_client.messages.create.return_value = tool_use_response()

    result = ai_generator.generate_response("What is MCP?", tools=tool_definitions)

    assert isinstance(result, str)


# --- System prompt / history ---------------------------------------------------------


def test_system_prompt_used_bare_without_history(ai_generator, mock_anthropic_client):
    ai_generator.generate_response("What is MCP?")

    assert call_params(mock_anthropic_client)["system"] == AIGenerator.SYSTEM_PROMPT


def test_conversation_history_is_appended_to_the_system_prompt(
    ai_generator, mock_anthropic_client
):
    ai_generator.generate_response("And lesson 2?", conversation_history="User: hi\nAssistant: hey")

    system = call_params(mock_anthropic_client)["system"]
    assert system.startswith(AIGenerator.SYSTEM_PROMPT)
    assert "Previous conversation:" in system
    assert "User: hi" in system


# --- Malformed / non-text responses --------------------------------------------------


def test_final_response_with_no_text_block_does_not_crash(
    ai_generator, mock_anthropic_client, tool_manager, tool_definitions
):
    """If max_tokens truncates the follow-up, or Claude emits another tool_use block,
    `content[0].text` raises and the request 500s. The caller must get a string."""
    mock_anthropic_client.messages.create.side_effect = [
        tool_use_response(),
        SimpleNamespace(
            stop_reason="tool_use", content=[tool_use_block("search_course_content", {"query": "x"})]
        ),
    ]

    result = ai_generator.generate_response(
        "What is MCP?", tools=tool_definitions, tool_manager=tool_manager
    )

    assert isinstance(result, str)


def test_empty_content_does_not_crash(ai_generator, mock_anthropic_client):
    mock_anthropic_client.messages.create.return_value = SimpleNamespace(
        stop_reason="end_turn", content=[]
    )

    result = ai_generator.generate_response("What is MCP?")

    assert isinstance(result, str)


def test_text_is_found_even_when_it_is_not_the_first_block(
    ai_generator, mock_anthropic_client
):
    mock_anthropic_client.messages.create.return_value = SimpleNamespace(
        stop_reason="end_turn",
        content=[
            SimpleNamespace(type="thinking", thinking="hmm"),
            text_block("The real answer."),
        ],
    )

    assert ai_generator.generate_response("What is MCP?") == "The real answer."


# --- API errors ----------------------------------------------------------------------


def test_authentication_error_becomes_a_readable_message(ai_generator, mock_anthropic_client):
    """A placeholder/expired ANTHROPIC_API_KEY must not surface as a bare 401."""
    mock_anthropic_client.messages.create.side_effect = anthropic.AuthenticationError(
        "invalid x-api-key",
        response=httpx.Response(401, request=httpx.Request("POST", "https://api.anthropic.com")),
        body=None,
    )

    with pytest.raises(RuntimeError, match="ANTHROPIC_API_KEY"):
        ai_generator.generate_response("What is MCP?")
