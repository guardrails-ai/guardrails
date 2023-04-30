import openai

import guardrails as gd

from .mock_llm_outputs import openai_completion_create, pydantic


def test_pydantic_with_reask(mocker):
    """Test that the entity extraction works with re-asking."""
    mocker.patch(
        "guardrails.llm_providers.openai_wrapper", new=openai_completion_create
    )

    guard = gd.Guard.from_rail_string(pydantic.RAIL_SPEC_WITH_REASK)
    _, final_output = guard(
        openai.Completion.create,
        engine="text-davinci-003",
        max_tokens=512,
        temperature=0.5,
        num_reasks=2,
    )

    # Assertions are made on the guard state object.
    assert final_output == pydantic.VALIDATED_OUTPUT_REASK_3

    guard_history = guard.guard_state.most_recent_call.history

    # Check that the guard state object has the correct number of re-asks.
    assert len(guard_history) == 3

    # For orginal prompt and output
    assert guard_history[0].prompt == gd.Prompt(pydantic.COMPILED_PROMPT)
    assert guard_history[0].output == pydantic.LLM_OUTPUT
    assert guard_history[0].validated_output == pydantic.VALIDATED_OUTPUT_REASK_1

    # For re-asked prompt and output
    assert guard_history[1].prompt == gd.Prompt(pydantic.COMPILED_PROMPT_REASK_1)
    assert guard_history[1].output == pydantic.LLM_OUTPUT_REASK_1
    assert guard_history[1].validated_output == pydantic.VALIDATED_OUTPUT_REASK_2

    # For re-asked prompt #2 and output #2
    assert guard_history[2].prompt == gd.Prompt(pydantic.COMPILED_PROMPT_REASK_2)
    assert guard_history[2].output == pydantic.LLM_OUTPUT_REASK_2
    assert guard_history[2].validated_output == pydantic.VALIDATED_OUTPUT_REASK_3
