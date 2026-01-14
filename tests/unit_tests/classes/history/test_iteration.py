from guardrails.classes.generic.stack import Stack
from guardrails.classes.history.inputs import Inputs
from guardrails.classes.history.iteration import Iteration
from guardrails.classes.history.outputs import Outputs
from guardrails.constants import error_status, not_run_status
from guardrails.llm_providers import LiteLLMCallable
from guardrails.classes.llm.llm_response import LLMResponse
from guardrails.classes.validation.validator_logs import ValidatorLogs
from guardrails.actions.reask import FieldReAsk
from guardrails.validator_base import FailResult


def test_empty_initialization():
    iteration = Iteration(
        call_id="mock-call",
        index=0,
    )

    assert iteration.inputs == Inputs()
    assert iteration.outputs == Outputs()
    assert iteration.logs == Stack()
    assert iteration.tokens_consumed is None
    assert iteration.prompt_tokens_consumed is None
    assert iteration.completion_tokens_consumed is None
    assert iteration.raw_output is None
    assert iteration.parsed_output is None
    assert iteration.validation_response is None
    assert iteration.guarded_output is None
    assert iteration.reasks == []
    assert iteration.validator_logs == []
    assert iteration.error is None
    assert iteration.failed_validations == []
    assert iteration.status == not_run_status
    assert iteration.rich_group is not None


def test_non_empty_initialization():
    # Inputs
    llm_api = LiteLLMCallable(text="Respond with a greeting.")
    llm_output = "Hello there!"
    messages = [
        {
            "role": "system",
            "content": "You are a greeting bot.",
        },
        {
            "role": "user",
            "content": "Respond with a ${greeting_type} greeting.",
        },
    ]
    prompt_params = {"greeting_type": "friendly"}
    num_reasks = 0
    metadata = {"some_meta_data": "doesn't actually matter"}
    full_schema_reask = False

    inputs = Inputs(
        llm_api=llm_api,
        llm_output=llm_output,
        messages=messages,
        prompt_params=prompt_params,
        num_reasks=num_reasks,
        metadata=metadata,
        full_schema_reask=full_schema_reask,
    )

    # Outputs
    validation_result = FailResult(
        outcome="fail",
        error_message="Should not include punctuation",
        fix_value="Hello there",
    )
    llm_response_info = LLMResponse(
        output="Hello there!", prompt_token_count=10, response_token_count=3
    )
    parsed_output = "Hello there!"
    guarded_output = "Hello there"
    reask = FieldReAsk(incorrect_value="Hello there!", fail_results=[validation_result], path=[])
    reasks = [reask]
    validator_logs = [
        ValidatorLogs(
            registered_name="no-punctuation",
            validator_name="no-punctuation",
            value_before_validation="Hello there!",
            validation_result=validation_result,
            value_after_validation="Hello there",
            property_path="$",
        )
    ]
    error = "Validation Failed!"
    outputs = Outputs(
        llm_response_info=llm_response_info,
        parsed_output=parsed_output,
        validation_response=reask,
        guarded_output=guarded_output,
        reasks=reasks,
        validator_logs=validator_logs,
        error=error,
    )

    iteration = Iteration(call_id="mock-call", index=0, inputs=inputs, outputs=outputs)

    assert iteration.inputs == inputs
    assert iteration.outputs == outputs
    assert iteration.logs == Stack()
    assert iteration.tokens_consumed == 13
    assert iteration.prompt_tokens_consumed == 10
    assert iteration.completion_tokens_consumed == 3
    assert iteration.raw_output == "Hello there!"
    assert iteration.parsed_output == "Hello there!"
    assert iteration.validation_response == reask
    assert iteration.guarded_output == "Hello there"
    assert iteration.reasks == reasks
    assert iteration.validator_logs == validator_logs
    assert iteration.error == error
    assert iteration.status == error_status
