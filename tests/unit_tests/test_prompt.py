"""Unit tests for prompt and instructions parsing."""

from string import Template

import pytest
from pydantic import BaseModel, Field

import guardrails as gd
from guardrails.prompt.instructions import Instructions
from guardrails.prompt.prompt import Prompt
from guardrails.utils.constants import constants
from guardrails.utils.prompt_utils import prompt_content_for_schema

INSTRUCTIONS = "\nYou are a helpful bot, who answers only with valid JSON\n"

PROMPT = "Extract a string from the text"

REASK_PROMPT = """
Please try that again, extract a string from the text
${xml_output_schema}
${previous_response}
"""

SIMPLE_RAIL_SPEC = f"""
<rail version="0.1">
<output>
    <string name="test_string" description="A string for testing." />
</output>
<messages>
    <message role="system">

{INSTRUCTIONS}

    </message>
    <message role="user">

{PROMPT}

    </message>
</messages>
</rail>
"""


RAIL_WITH_PARAMS = """
<rail version="0.1">
<output>
    <string name="test_string" description="A string for testing." />
</output>
<messages>
    <message role="system">

${user_instructions}

    </message>
    <message role="user">

${user_prompt}

    </message>
</messages>
</rail>
"""


RAIL_WITH_FORMAT_INSTRUCTIONS = """
<rail version="0.1">
<output>
    <string name="test_string" description="A string for testing." />
</output>
<messages>
    <message role="system">

You are a helpful bot, who answers only with valid JSON

    </message>
    <message role="user">

Extract a string from the text

${gr.complete_json_suffix_v2}
    </message>
</messages>
</rail>
"""

RAIL_WITH_OLD_CONSTANT_SCHEMA = """
<rail version="0.1">
<output>
    <string name="test_string" description="A string for testing." />
</output>
<messages>
    <message role="system">

You are a helpful bot, who answers only with valid JSON

    </message>
    <message role="user">

Extract a string from the text
@gr.complete_json_suffix_v2
    </message>
</messages>
</rail>
"""

RAIL_WITH_REASK_MESSAGES_USER = """
<rail version="0.1">
<output>
    <string name="test_string" description="A string for testing." />
</output>
<messages>
    <message role="system">

You are a helpful bot, who answers only with valid JSON

    </message>
    <message role="user">
${gr.complete_json_suffix_v2}
    </message>
</messages>

<reask_messages>
<message role="user">
Please try that again, extract a string from the text
${xml_output_schema}
${previous_response}
</message>
</reask_messages>
</rail>
"""

RAIL_WITH_REASK_MESSAGES_USER_AND_SYSTEM = """
<rail version="0.1">
<output>
    <string name="test_string" description="A string for testing." />
</output>
<messages>
    <message role="system">

You are a helpful bot, who answers only with valid JSON

    </message>
    <message role="user">

Extract a string from the text

${gr.complete_json_suffix_v2}
    </message>
</messages>

<reask_messages>
<message role="user">
Please try that again, extract a string from the text
${xml_output_schema}
${previous_response}
</message>
<message role="system">
You are a helpful bot, who answers only with valid JSON
</message>
</reask_messages>
</rail>
"""


def test_parse_prompt():
    """Test parsing a prompt."""
    guard = gd.Guard.for_rail_string(SIMPLE_RAIL_SPEC)

    # Strip both, raw and parsed, to be safe
    instructions = Instructions(guard._exec_opts.messages[0]["content"])
    assert instructions.format().source.strip() == INSTRUCTIONS.strip()
    prompt = Prompt(guard._exec_opts.messages[1]["content"])
    assert prompt.format().source.strip() == PROMPT.strip()


def test_instructions_with_params():
    """Test a guard with instruction parameters."""
    guard = gd.Guard.for_rail_string(RAIL_WITH_PARAMS)

    user_instructions = "A useful system message."
    user_prompt = "A useful prompt."

    instructions = Instructions(guard._exec_opts.messages[0]["content"])
    assert (
        instructions.format(user_instructions=user_instructions).source.strip()
        == user_instructions.strip()
    )
    prompt = Prompt(guard._exec_opts.messages[1]["content"])
    assert prompt.format(user_prompt=user_prompt).source.strip() == user_prompt.strip()


@pytest.mark.parametrize(
    "rail,var_names",
    [
        (SIMPLE_RAIL_SPEC, []),
        (RAIL_WITH_PARAMS, ["user_prompt"]),
    ],
)
def test_variable_names(rail, var_names):
    """Test extracting variable names from a prompt."""
    guard = gd.Guard.for_rail_string(rail)

    prompt = Prompt(guard._exec_opts.messages[1]["content"])

    assert prompt.variable_names == var_names


def test_format_instructions():
    """Test extracting format instructions from a prompt."""
    guard = gd.Guard.for_rail_string(RAIL_WITH_FORMAT_INSTRUCTIONS)

    output_schema = prompt_content_for_schema(
        guard._output_type,
        guard.output_schema.to_dict(),
        validator_map=guard._validator_map,
        json_path="$",
    )

    expected_instructions = (
        Template(constants["complete_json_suffix_v2"])
        .safe_substitute(output_schema=output_schema)
        .rstrip()
    )
    prompt = Prompt(guard._exec_opts.messages[1]["content"], output_schema=output_schema)
    assert prompt.format_instructions.rstrip() == expected_instructions


def test_reask_messages_user():
    guard = gd.Guard.for_rail_string(RAIL_WITH_REASK_MESSAGES_USER)
    assert guard._exec_opts.reask_messages[0]["content"] == REASK_PROMPT


def test_reask_messages_user_and_system():
    guard = gd.Guard.for_rail_string(RAIL_WITH_REASK_MESSAGES_USER_AND_SYSTEM)
    assert guard._exec_opts.reask_messages[1]["content"] == INSTRUCTIONS


@pytest.mark.parametrize(
    "prompt_str,final_prompt",
    [
        (
            "Dummy prompt. ${gr.complete_json_suffix_v2}",
            f"Dummy prompt. {constants['complete_json_suffix_v2']}",
        ),
        ("Dummy prompt. some@email.com", "Dummy prompt. some@email.com"),
    ],
)
def test_substitute_constants(prompt_str, final_prompt):
    """Test substituting constants in a prompt."""
    prompt = gd.Prompt(prompt_str)
    assert prompt.source == final_prompt


class TestResponse(BaseModel):
    grade: int = Field(description="The grade of the response")


def test_gr_prefixed_prompt_item_passes():
    # From pydantic:
    prompt = """Give me a response to ${grade}"""
    messages = [
        {
            "role": "user",
            "content": prompt,
        }
    ]

    guard = gd.Guard.for_pydantic(output_class=TestResponse, messages=messages)
    prompt = Prompt(guard._exec_opts.messages[0]["content"])
    assert len(prompt.variable_names) == 1


def test_gr_dot_prefixed_prompt_item_fails():
    with pytest.raises(Exception):
        # From pydantic:
        prompt = """Give me a response to ${gr.ade}"""
        Prompt(prompt)


def test_escape():
    prompt_string = (
        'My prompt with a some sample json { "a" : 1 } and a {f_var} and a'
        " ${safe_var}. Also an incomplete brace {."
    )
    prompt = Prompt(prompt_string)

    assert prompt.source == prompt_string
    assert prompt.escape() == (
        'My prompt with a some sample json {{ "a" : 1 }} and a {{f_var}} and a'
        " ${safe_var}. Also an incomplete brace {{."
    )
