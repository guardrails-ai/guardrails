import guardrails as gd
from guardrails.classes.llm.llm_response import LLMResponse

import tests.integration_tests.test_assets.validators  # noqa

from .test_assets import python_rail


def test_multi_reask(mocker):
    """Test that parallel reasking works."""
    mock_invoke_llm = mocker.patch("guardrails.llm_providers.LiteLLMCallable._invoke_llm")
    mock_invoke_llm.side_effect = [
        LLMResponse(
            output=python_rail.VALIDATOR_PARALLELISM_RESPONSE_1,
            prompt_token_count=123,
            response_token_count=1234,
        ),
        LLMResponse(
            output=python_rail.VALIDATOR_PARALLELISM_RESPONSE_2,
            prompt_token_count=123,
            response_token_count=1234,
        ),
        LLMResponse(
            output=python_rail.VALIDATOR_PARALLELISM_RESPONSE_3,
            prompt_token_count=123,
            response_token_count=1234,
        ),
    ]

    guard = gd.Guard.for_rail_string(python_rail.RAIL_SPEC_WITH_VALIDATOR_PARALLELISM)

    guard(
        model="text-davinci-003",
        num_reasks=5,
    )

    # Assertions are made on the guard state object.
    # assert final_output == python_rail

    call = guard.history.first

    assert len(call.iterations) == 3

    assert call.compiled_messages[0]["content"] == python_rail.VALIDATOR_PARALLELISM_PROMPT_1
    assert call.raw_outputs.first == python_rail.VALIDATOR_PARALLELISM_RESPONSE_1
    assert call.iterations.first.validation_response == python_rail.VALIDATOR_PARALLELISM_REASK_1

    assert call.reask_messages[0][1]["content"] == python_rail.VALIDATOR_PARALLELISM_PROMPT_2
    assert call.raw_outputs.at(1) == python_rail.VALIDATOR_PARALLELISM_RESPONSE_2
    assert call.iterations.at(1).validation_response == python_rail.VALIDATOR_PARALLELISM_REASK_2

    assert call.reask_messages[1][1]["content"] == python_rail.VALIDATOR_PARALLELISM_PROMPT_3
    assert call.raw_outputs.last == python_rail.VALIDATOR_PARALLELISM_RESPONSE_3
    # The output here fails some validators but passes others.
    # Since those that it fails in the end are noop fixes, validation fails.
    assert call.validation_response == python_rail.VALIDATOR_PARALLELISM_RESPONSE_3
    assert call.guarded_output is not None and isinstance(call.guarded_output, str)
    assert call.status == "fail"
