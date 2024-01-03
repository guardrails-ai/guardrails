import pytest

from guardrails.classes.history.outputs import Outputs
from guardrails.constants import error_status, fail_status, not_run_status, pass_status
from guardrails.utils.llm_response import LLMResponse
from guardrails.utils.logs_utils import ValidatorLogs
from guardrails.utils.reask_utils import ReAsk
from guardrails.validator_base import FailResult, PassResult


def test_empty_initialization():
    empty_outputs = Outputs()

    assert empty_outputs.llm_response_info is None
    assert empty_outputs.parsed_output is None
    assert empty_outputs.validation_output is None
    assert empty_outputs.validated_output is None
    assert empty_outputs.reasks == []
    assert empty_outputs.validator_logs == []
    assert empty_outputs.error is None
    assert empty_outputs.failed_validations == []
    assert empty_outputs.status == not_run_status


def test_non_empty_initialization():
    validation_result = FailResult(
        outcome="fail",
        error_message="Should not include punctuation",
        fix_value="Hello there",
    )
    llm_response_info = LLMResponse(
        output="Hello there!", prompt_token_count=10, response_token_count=3
    )
    parsed_output = "Hello there!"
    validated_output = "Hello there"
    reasks = [ReAsk(incorrect_value="Hello there!", fail_results=[validation_result])]
    validator_logs = [
        ValidatorLogs(
            validator_name="no-punctuation",
            value_before_validation="Hello there!",
            validation_result=validation_result,
            value_after_validation="Hello there",
            property_path="$",
        )
    ]
    error = "Validation Failed!"
    non_empty_outputs = Outputs(
        llm_response_info=llm_response_info,
        parsed_output=parsed_output,
        validated_output=validated_output,
        reasks=reasks,
        validator_logs=validator_logs,
        error=error,
    )

    assert non_empty_outputs.llm_response_info is not None
    assert non_empty_outputs.llm_response_info == llm_response_info
    assert non_empty_outputs.parsed_output is not None
    assert non_empty_outputs.parsed_output == parsed_output
    assert non_empty_outputs.validated_output is not None
    assert non_empty_outputs.validated_output == validated_output
    assert non_empty_outputs.reasks != []
    assert non_empty_outputs.reasks == reasks
    assert non_empty_outputs.validator_logs != []
    assert non_empty_outputs.validator_logs == validator_logs
    assert non_empty_outputs.error is not None
    assert non_empty_outputs.error == error
    assert non_empty_outputs.failed_validations == validator_logs
    assert non_empty_outputs.status == error_status


fixable_fail_result = FailResult(
    outcome="fail",
    error_message="Should not include punctuation",
    fix_value="Hello there",
)
non_fixable_fail_result = FailResult(
    outcome="fail",
    error_message="Should not include punctuation",
)


@pytest.mark.parametrize(
    "outputs,expected_result",
    [
        (Outputs(), True),
        (Outputs(llm_response_info=LLMResponse(output="Hello there!")), False),
        (Outputs(parsed_output="Hello there!"), False),
        (Outputs(parsed_output="Hello there!"), False),
        (Outputs(validated_output="Hello there"), False),
        (
            Outputs(
                reasks=[
                    ReAsk(
                        incorrect_value="Hello there!",
                        fail_results=[fixable_fail_result],
                    )
                ]
            ),
            False,
        ),
        (
            Outputs(
                validator_logs=[
                    ValidatorLogs(
                        validator_name="no-punctuation",
                        value_before_validation="Hello there!",
                        validation_result=fixable_fail_result,
                        value_after_validation="Hello there",
                        property_path="$",
                    )
                ]
            ),
            False,
        ),
        (Outputs(error="Validation Failed!"), False),
    ],
)
def test__all_empty(outputs: Outputs, expected_result: bool):
    are_outputs_empty = outputs._all_empty()

    assert are_outputs_empty == expected_result


def test_failed_validations():
    validator_logs = [
        ValidatorLogs(
            validator_name="no-punctuation",
            value_before_validation="Hello there!",
            validation_result=fixable_fail_result,
            value_after_validation="Hello there",
            property_path="$",
        ),
        ValidatorLogs(
            validator_name="valid-length",
            value_before_validation="Hello there!",
            validation_result=PassResult(),
            value_after_validation="Hello there!",
            property_path="$",
        ),
    ]
    outputs = Outputs(validator_logs=validator_logs)

    assert outputs.failed_validations == [validator_logs[0]]


@pytest.mark.parametrize(
    "outputs,expected_status",
    [
        (Outputs(), not_run_status),
        (Outputs(error="Validations Failed!"), error_status),
        (
            Outputs(
                validator_logs=[
                    ValidatorLogs(
                        validator_name="no-punctuation",
                        value_before_validation="Hello there!",
                        validation_result=non_fixable_fail_result,
                        value_after_validation="Hello there",
                        property_path="$",
                    )
                ],
                reasks=[
                    ReAsk(
                        incorrect_value="Hello there!",
                        fail_results=[non_fixable_fail_result],
                    )
                ],
            ),
            fail_status,
        ),
        (Outputs(validator_logs=[], validated_output="Hello there!"), pass_status),
        (
            Outputs(
                validation_output=ReAsk(
                    incorrect_value="Hello there!",
                    fail_results=[non_fixable_fail_result],
                ),
            ),
            fail_status,
        ),
    ],
)
def test_status(outputs: Outputs, expected_status: str):
    status = outputs.status

    assert status == expected_status
