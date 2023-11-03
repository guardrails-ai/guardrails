"""Unit tests for prompt and instructions parsing."""

from string import Template
from unittest import mock

import pytest
from pydantic import BaseModel, Field

import guardrails as gd
from guardrails.prompt.instructions import Instructions
from guardrails.prompt.prompt import Prompt
from guardrails.utils.constants import constants

INSTRUCTIONS = "\nYou are a helpful bot, who answers only with valid JSON\n"

PROMPT = "Extract a string from the text"

REASK_PROMPT = """
Please try that again, extract a string from the text
${output_schema}
${previous_response}
"""

SIMPLE_RAIL_SPEC = f"""
<rail version="0.1">
<output>
    <string name="test_string" description="A string for testing." />
</output>
<instructions>

{INSTRUCTIONS}

</instructions>

<prompt>

{PROMPT}

</prompt>
</rail>
"""


RAIL_WITH_PARAMS = """
<rail version="0.1">
<output>
    <string name="test_string" description="A string for testing." />
</output>
<instructions>

${user_instructions}

</instructions>

<prompt>

${user_prompt}

</prompt>
</rail>
"""


RAIL_WITH_FORMAT_INSTRUCTIONS = """
<rail version="0.1">
<output>
    <string name="test_string" description="A string for testing." />
</output>
<instructions>

You are a helpful bot, who answers only with valid JSON

</instructions>

<prompt>

Extract a string from the text

${gr.complete_json_suffix_v2}
</prompt>
</rail>
"""

RAIL_WITH_OLD_CONSTANT_SCHEMA = """
<rail version="0.1">
<output>
    <string name="test_string" description="A string for testing." />
</output>
<instructions>

You are a helpful bot, who answers only with valid JSON

</instructions>

<prompt>

Extract a string from the text
@gr.complete_json_suffix_v2
</prompt>
</rail>
"""

RAIL_WITH_REASK_PROMPT = """
<rail version="0.1">
<output>
    <string name="test_string" description="A string for testing." />
</output>
<instructions>

You are a helpful bot, who answers only with valid JSON

</instructions>

<prompt>
${gr.complete_json_suffix_v2}
</prompt>
<reask_prompt>
Please try that again, extract a string from the text
${output_schema}
${previous_response}
</reask_prompt>
</rail>
"""

RAIL_WITH_REASK_INSTRUCTIONS = """
<rail version="0.1">
<output>
    <string name="test_string" description="A string for testing." />
</output>
<instructions>

You are a helpful bot, who answers only with valid JSON

</instructions>

<prompt>

Extract a string from the text

${gr.complete_json_suffix_v2}
</prompt>
<reask_prompt>
Please try that again, extract a string from the text
${output_schema}
${previous_response}
</reask_prompt>
<reask_instructions>
You are a helpful bot, who answers only with valid JSON
</reask_instructions>
</rail>
"""


def test_parse_prompt():
    """Test parsing a prompt."""
    guard = gd.Guard.from_rail_string(SIMPLE_RAIL_SPEC)

    # Strip both, raw and parsed, to be safe
    assert guard.instructions.format().source.strip() == INSTRUCTIONS.strip()
    assert guard.prompt.format().source.strip() == PROMPT.strip()


def test_instructions_with_params():
    """Test a guard with instruction parameters."""
    guard = gd.Guard.from_rail_string(RAIL_WITH_PARAMS)

    user_instructions = "A useful system message."
    user_prompt = "A useful prompt."

    assert (
        guard.instructions.format(user_instructions=user_instructions).source.strip()
        == user_instructions.strip()
    )
    assert (
        guard.prompt.format(user_prompt=user_prompt).source.strip()
        == user_prompt.strip()
    )


@pytest.mark.parametrize(
    "rail,var_names",
    [
        (SIMPLE_RAIL_SPEC, []),
        (RAIL_WITH_PARAMS, ["user_prompt"]),
    ],
)
def test_variable_names(rail, var_names):
    """Test extracting variable names from a prompt."""
    guard = gd.Guard.from_rail_string(rail)

    assert guard.prompt.variable_names == var_names


def test_format_instructions():
    """Test extracting format instructions from a prompt."""
    guard = gd.Guard.from_rail_string(RAIL_WITH_FORMAT_INSTRUCTIONS)
    output_schema = guard.rail.output_schema.transpile()
    expected_instructions = (
        Template(constants["complete_json_suffix_v2"])
        .safe_substitute(output_schema=output_schema)
        .rstrip()
    )

    assert guard.prompt.format_instructions.rstrip() == expected_instructions


def test_reask_prompt():
    guard = gd.Guard.from_rail_string(RAIL_WITH_REASK_PROMPT)
    assert guard.output_schema.reask_prompt_template == Prompt(REASK_PROMPT)


def test_reask_instructions():
    guard = gd.Guard.from_rail_string(RAIL_WITH_REASK_INSTRUCTIONS)
    assert guard.output_schema._reask_instructions_template == Instructions(
        INSTRUCTIONS
    )


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


# TODO: Deprecate when we can confirm migration off the old, non-namespaced standard
@pytest.mark.parametrize(
    "text, is_old_schema",
    [
        (RAIL_WITH_OLD_CONSTANT_SCHEMA, True),  # Test with a single match
        (
            RAIL_WITH_FORMAT_INSTRUCTIONS,
            False,
        ),  # Test with no matches/correct namespacing
    ],
)
def test_uses_old_constant_schema(text, is_old_schema):
    with mock.patch("warnings.warn") as warn_mock:
        guard = gd.Guard.from_rail_string(text)
        assert guard.prompt.uses_old_constant_schema(text) == is_old_schema
        if is_old_schema:
            # we only check for the warning when we have an older schema
            warn_mock.assert_called_once_with(
                """It appears that you are using an old schema for gaurdrails\
 variables, follow the new namespaced convention documented here:\
 https://docs.guardrailsai.com/0-2-migration/\
"""
            )


class TestResponse(BaseModel):
    grade: int = Field(description="The grade of the response")


def test_gr_prefixed_prompt_item_passes():
    # From pydantic:
    prompt = """Give me a response to ${grade}"""

    guard = gd.Guard.from_pydantic(output_class=TestResponse, prompt=prompt)
    assert len(guard.prompt.variable_names) == 1


def test_gr_dot_prefixed_prompt_item_fails():
    with pytest.raises(Exception):
        # From pydantic:
        prompt = """Give me a response to ${gr.ade}"""
        gd.Guard.from_pydantic(output_class=TestResponse, prompt=prompt)


def test_escape():
    prompt_string = (
        'My prompt with a some sample json { "a" : 1 } and a {f_var} and a ${safe_var}'
    )
    prompt = Prompt(prompt_string)

    escaped_prompt = prompt.escape()

    assert prompt.source == prompt_string
    assert (
        escaped_prompt
        == 'My prompt with a some sample json {{ "a" : 1 }} and a {{f_var}} and a ${safe_var}'  # noqa
    )
