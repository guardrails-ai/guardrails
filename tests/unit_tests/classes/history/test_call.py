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
from guardrails.classes.llm.llm_response import LLMResponse
from guardrails.classes.validation.validator_logs import ValidatorLogs
from guardrails.actions.reask import ReAsk
from guardrails.validator_base import FailResult, PassResult


def test_empty_initialization():
    call = Call()

    assert call.iterations == Stack()
    assert call.inputs == CallInputs()
    assert call.prompt is None
    assert call.prompt_params is None
    assert call.compiled_prompt is None
    assert call.reask_prompts == Stack()
    assert call.instructions is None
    assert call.compiled_instructions is None
    assert call.reask_instructions == Stack()
    assert call.logs == Stack()
    assert call.tokens_consumed is None
    assert call.prompt_tokens_consumed is None
    assert call.completion_tokens_consumed is None
    assert call.raw_outputs == Stack()
    assert call.parsed_outputs == Stack()
    assert call.validation_response is None
    assert call.fixed_output is None
    assert call.guarded_output is None
    assert call.reasks == Stack()
    assert call.validator_logs == Stack()
    assert call.error is None
    assert call.failed_validations == Stack()
    assert call.status == not_run_status
    # FIXME: how to do shallow comparison?
    # assert call.tree == Tree("Logs")
    assert call.tree is not None


def test_non_empty_initialization():
    # Call input
    def custom_llm():
        return "Hello there!"

    llm_api = custom_llm
    prompt = "Respond with a ${greeting_type} greeting."
    instructions = "You are a greeting bot."
    args = ["arg1"]
    kwargs = {"kwarg1": 1}
    prompt_params = {"greeting_type": "friendly"}

    call_inputs = CallInputs(
        llm_api=llm_api,
        prompt=prompt,
        instructions=instructions,
        prompt_params=prompt_params,
        args=args,
        kwargs=kwargs,
    )

    # First Iteration Inputs
    iter_llm_api = ArbitraryCallable(llm_api=llm_api)
    llm_output = "Hello there!"
    instructions = Instructions(source=instructions)
    iter_prompt = Prompt(source=prompt)
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
    first_validator_log = ValidatorLogs(
        registered_name="no-punctuation",
        validator_name="no-punctuation",
        value_before_validation="Hello there!",
        validation_result=first_validation_result,
        value_after_validation="Hello there",
        property_path="$",
    )
    first_validator_logs = [first_validator_log]
    first_outputs = Outputs(
        llm_response_info=first_llm_response_info,
        parsed_output=first_parsed_output,
        validated_output=first_validated_output,
        reasks=first_reasks,
        validator_logs=first_validator_logs,
    )

    first_iteration = Iteration(inputs=inputs, outputs=first_outputs)

    second_iter_prompt = Prompt(source="That wasn't quite right. Try again.")

    second_inputs = Inputs(
        llm_api=iter_llm_api,
        llm_output=llm_output,
        instructions=instructions,
        prompt=second_iter_prompt,
        num_reasks=num_reasks,
        metadata=metadata,
        full_schema_reask=full_schema_reask,
    )

    second_llm_response_info = LLMResponse(
        output="Hello there", prompt_token_count=10, response_token_count=3
    )
    second_parsed_output = "Hello there"
    second_validated_output = "Hello there"
    second_reasks = []
    second_validator_log = ValidatorLogs(
        registered_name="no-punctuation",
        validator_name="no-punctuation",
        value_before_validation="Hello there",
        validation_result=PassResult(),
        value_after_validation="Hello there",
        property_path="$",
    )
    second_validator_logs = [second_validator_log]
    second_outputs = Outputs(
        llm_response_info=second_llm_response_info,
        parsed_output=second_parsed_output,
        validation_response="Hello there",
        validated_output=second_validated_output,
        reasks=second_reasks,
        validator_logs=second_validator_logs,
    )

    second_iteration = Iteration(inputs=second_inputs, outputs=second_outputs)

    iterations: Stack[Iteration] = Stack(first_iteration, second_iteration)

    call = Call(inputs=call_inputs, iterations=iterations)

    assert call.iterations == iterations
    assert isinstance(call.iterations, Stack) is True
    assert call.inputs == call_inputs

    assert call.prompt == prompt
    assert call.prompt_params == prompt_params
    assert call.compiled_prompt == "Respond with a friendly greeting."
    assert call.reask_prompts == Stack(second_iter_prompt.source)
    assert call.instructions == instructions.source
    assert call.compiled_instructions == instructions.source
    assert call.reask_instructions == Stack(instructions.source)

    # TODO: Test this in the integration tests
    assert call.logs == []

    assert call.tokens_consumed == 26
    assert call.prompt_tokens_consumed == 20
    assert call.completion_tokens_consumed == 6

    assert call.raw_outputs == Stack("Hello there!", "Hello there")
    assert call.parsed_outputs == Stack("Hello there!", "Hello there")
    assert call.validation_response == "Hello there"
    assert call.fixed_output == "Hello there"
    assert call.guarded_output == "Hello there"
    assert call.reasks == Stack()
    assert call.validator_logs == Stack(first_validator_log, second_validator_log)
    assert call.error is None
    assert call.failed_validations == Stack(first_validator_log)
    assert call.status == pass_status
    # TODO: How to do shallow comparison
    # assert call.tree == "something"
    assert call.tree is not None
