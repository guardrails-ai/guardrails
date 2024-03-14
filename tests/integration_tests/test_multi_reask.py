import guardrails as gd
from guardrails.utils.openai_utils import get_static_openai_create_func

from .mock_llm_outputs import MockOpenAICallable
from .test_assets import python_rail


def test_multi_reask(mocker):
    """Test that parallel reasking works."""
    mocker.patch("guardrails.llm_providers.OpenAICallable", new=MockOpenAICallable)

    guard = gd.Guard.from_rail_string(python_rail.RAIL_SPEC_WITH_VALIDATOR_PARALLELISM)

    guard(
        llm_api=get_static_openai_create_func(),
        engine="text-davinci-003",
        num_reasks=5,
    )

    # Assertions are made on the guard state object.
    # assert final_output == python_rail

    call = guard.history.first

    assert len(call.iterations) == 3

    assert call.compiled_prompt == python_rail.VALIDATOR_PARALLELISM_PROMPT_1
    assert call.raw_outputs.first == python_rail.VALIDATOR_PARALLELISM_RESPONSE_1
    assert (
        call.iterations.first.validation_response
        == python_rail.VALIDATOR_PARALLELISM_REASK_1
    )

    assert call.reask_prompts.first == python_rail.VALIDATOR_PARALLELISM_PROMPT_2
    assert call.raw_outputs.at(1) == python_rail.VALIDATOR_PARALLELISM_RESPONSE_2
    assert (
        call.iterations.at(1).validation_response
        == python_rail.VALIDATOR_PARALLELISM_REASK_2
    )

    assert call.reask_prompts.last == python_rail.VALIDATOR_PARALLELISM_PROMPT_3
    assert call.raw_outputs.last == python_rail.VALIDATOR_PARALLELISM_RESPONSE_3
    # The output here fails some validators but passes others.
    # Since those that it fails in the end are noop fixes, validation fails.
    assert call.validation_response == python_rail.VALIDATOR_PARALLELISM_RESPONSE_3
    assert call.validated_output is None
    assert call.status == "fail"
