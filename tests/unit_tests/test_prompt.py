"""Unit tests for prompt and instructions parsing."""

import guardrails as gd

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


def test_parse_prompt():
    """Test parsing a prompt."""
    guard = gd.Guard.from_rail_string(SIMPLE_RAIL_SPEC)

    # Strip both, raw and parsed, to be safe
    assert guard.instructions.format().strip() == INSTRUCTIONS.strip()
    assert guard.base_prompt.format().strip() == PROMPT.strip()


def test_instructions_with_params():
    """Test a guard with instruction parameters."""
    guard = gd.Guard.from_rail_string(RAIL_WITH_PARAMS)

    user_instructions = "A useful system message."
    user_prompt = "A useful prompt."

    assert (
        guard.instructions.format(user_instructions=user_instructions).strip()
        == user_instructions.strip()
    )
    assert (
        guard.base_prompt.format(user_prompt=user_prompt).strip() == user_prompt.strip()
    )
