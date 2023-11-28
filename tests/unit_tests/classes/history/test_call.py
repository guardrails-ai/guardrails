from guardrails.classes.generic.stack import Stack
from guardrails.classes.history.call import Call
from guardrails.classes.history.call_inputs import CallInputs
from guardrails.classes.history.inputs import Inputs
from guardrails.classes.history.iteration import Iteration
from guardrails.classes.history.outputs import Outputs
from guardrails.constants import not_run_status, pass_status
from guardrails.llm_providers import ArbitraryCallable
from guardrails.prompt.instructions import Instructions
from guardrails.prompt.prompt import Prompt
from guardrails.utils.llm_response import LLMResponse
from guardrails.utils.logs_utils import ValidatorLogs
from guardrails.utils.reask_utils import ReAsk
from guardrails.validator_base import FailResult, PassResult


def test_empty_initialization():
    call = Call()

    # Overrides and additional properties
    assert call.iterations == Stack()
    assert isinstance(call.iterations, Stack) is True
    assert call.inputs == CallInputs()
    call_outputs = call.outputs
    # print("call.outputs: ", call_outputs.json())
    print("Outputs(): ", Outputs())
    assert call_outputs == Outputs()
    assert call.tokens_consumed is None
    assert call.prompt_tokens_consumed is None
    assert call.completion_tokens_consumed is None
    assert call.status == not_run_status

    # Inherited properties
    assert call.raw_output is None
    assert call.parsed_output is None
    assert call.validated_output is None
    assert call.reasks == []
    assert call.validator_logs == []
    assert call.error is None
    assert call.failed_validations == []


def test_non_empty_initialization():

    # Call input
    def custom_llm():
        return "Hello there!"

    llm_api = custom_llm
    prompt = "Respond with a friendly greeting."
    instructions = "You are a greeting bot."
    args = ["arg1"]
    kwargs = {"kwarg1": 1}

    call_inputs = CallInputs(
        llm_api=llm_api,
        prompt=prompt,
        instructions=instructions,
        args=args,
        kwargs=kwargs,
    )

    # First Iteration Inputs
    iter_llm_api = ArbitraryCallable(llm_api=llm_api)
    llm_output = "Hello there!"
    instructions = Instructions(source="You are a greeting bot.")
    iter_prompt = Prompt(source="Respond with a ${greeting_type} greeting.")
    prompt_params = {"greeting_type": "friendly"}
    num_reasks = 0
    metadata = {"some_meta_data": "doesn't actually matter"}
    full_schema_reask = False

    inputs = Inputs(
        llm_api=iter_llm_api,
        llm_output=llm_output,
        instructions=instructions,
        prompt=iter_prompt,
        prompt_params=prompt_params,
        num_reasks=num_reasks,
        metadata=metadata,
        full_schema_reask=full_schema_reask,
    )

    # Outputs
    first_validation_result = FailResult(
        outcome="fail",
        error_message="Should not include punctuation",
        fix_value="Hello there",
    )
    first_llm_response_info = LLMResponse(
        output="Hello there!", prompt_token_count=10, response_token_count=3
    )
    first_parsed_output = "Hello there!"
    first_validated_output = "Hello there"
    first_reasks = [
        ReAsk(incorrect_value="Hello there!", fail_results=[first_validation_result])
    ]
    first_validator_logs = [
        ValidatorLogs(
            validator_name="no-punctuation",
            value_before_validation="Hello there!",
            validation_result=first_validation_result,
            value_after_validation="Hello there",
        )
    ]
    first_outputs = Outputs(
        llm_response_info=first_llm_response_info,
        parsed_output=first_parsed_output,
        validated_output=first_validated_output,
        reasks=first_reasks,
        validator_logs=first_validator_logs,
    )

    first_iteration = Iteration(inputs=inputs, outputs=first_outputs)

    second_llm_response_info = LLMResponse(
        output="Hello there", prompt_token_count=10, response_token_count=3
    )
    second_parsed_output = "Hello there"
    second_validated_output = "Hello there"
    second_reasks = []
    second_validator_logs = [
        ValidatorLogs(
            validator_name="no-punctuation",
            value_before_validation="Hello there",
            validation_result=PassResult(),
            value_after_validation="Hello there",
        )
    ]
    second_outputs = Outputs(
        llm_response_info=second_llm_response_info,
        parsed_output=second_parsed_output,
        validated_output=second_validated_output,
        reasks=second_reasks,
        validator_logs=second_validator_logs,
    )

    second_iteration = Iteration(inputs=inputs, outputs=second_outputs)

    iterations: Stack[Iteration] = Stack(first_iteration, second_iteration)

    call = Call(inputs=call_inputs, iterations=iterations)

    assert call.iterations == iterations
    assert isinstance(call.iterations, Stack) is True
    assert call.inputs == call_inputs
    assert call.outputs == second_iteration.outputs
    assert call.tokens_consumed == 26
    assert call.prompt_tokens_consumed == 20
    assert call.completion_tokens_consumed == 6
    assert call.status == pass_status
