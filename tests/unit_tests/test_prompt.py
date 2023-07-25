"""Unit tests for prompt and instructions parsing."""

import pytest

import guardrails as gd
from guardrails.utils.constants import constants

INSTRUCTIONS = "You are a helpful bot, who answers only with valid JSON"

PROMPT = "Extract a string from the text"

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

{{user_instructions}}

</instructions>

<prompt>

{{user_prompt}}

</prompt>
</rail>
"""


RAIL_WITH_FORMAT_INSTRUCTIONS = f"""
<rail version="0.1">
<output>
    <string name="test_string" description="A string for testing." />
</output>
<instructions>

{INSTRUCTIONS}

</instructions>

<prompt>

{PROMPT}

@complete_json_suffix_v2
</prompt>
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
        constants["complete_json_suffix_v2"]
        .format(output_schema=output_schema)
        .rstrip()
    )

    assert guard.prompt.format_instructions.rstrip() == expected_instructions


@pytest.mark.parametrize(
    "prompt_str,final_prompt",
    [
        (
            "Dummy prompt. @complete_json_suffix_v2",
            f"Dummy prompt. {constants['complete_json_suffix_v2']}",
        ),
        ("Dummy prompt. some@email.com", "Dummy prompt. some@email.com"),
    ],
)
def test_substitute_constants(prompt_str, final_prompt):
    """Test substituting constants in a prompt."""
    prompt = gd.Prompt(prompt_str)
    substituted_prompt = prompt.substitute_constants(prompt.source)
    assert substituted_prompt == final_prompt
