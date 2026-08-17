from guardrails.classes.validation.validation_summary import ValidationSummary
from guardrails.classes.validation.validator_logs import ValidatorLogs
from guardrails_ai.types import FailResult, Outcome


def test_validation_summary_includes_before_and_after_values():
    validator_log = ValidatorLogs(
        validatorName="MockValidator",
        registeredName="mock-validator",
        valueBeforeValidation="original value",
        valueAfterValidation="fixed value",
        propertyPath="$",
        validationResult=FailResult(
            outcome=Outcome.FAIL,
            errorMessage="Value failed validation.",
            fixValue="fixed value",
        ),
    )

    summaries = ValidationSummary.from_validator_logs_only_fails([validator_log])

    assert len(summaries) == 1
    summary = summaries[0]

    assert summary.validator_name == "MockValidator"
    assert summary.property_path == "$"
    assert summary.failure_reason == "Value failed validation."
    assert summary.value_before_validation == "original value"
    assert summary.value_after_validation == "fixed value"

    dumped = summary.model_dump(by_alias=True)

    assert dumped["valueBeforeValidation"] == "original value"
    assert dumped["valueAfterValidation"] == "fixed value"
