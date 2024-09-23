# TODO Temp to update once generated class is in
from typing import List, Optional

from guardrails.classes.generic.arbitrary_model import ArbitraryModel
from guardrails.classes.validation.validation_result import ErrorSpan, FailResult
from guardrails.classes.validation.validator_logs import ValidatorLogs


class ValidationSummary(ArbitraryModel):
    validator_name: str
    validator_status: str
    failure_reason: Optional[str]
    error_spans: Optional[List["ErrorSpan"]] = []
    property_path: Optional[str]

    @staticmethod
    def from_validator_logs(
        validator_logs: List[ValidatorLogs],
    ) -> List["ValidationSummary"]:
        summaries = []
        for log in validator_logs:
            validation_result = log.validation_result
            is_fail_result = isinstance(validation_result, FailResult)
            failure_reason = validation_result.error_message if is_fail_result else None
            error_spans = validation_result.error_spans if is_fail_result else []
            summaries.append(
                ValidationSummary(
                    validator_name=log.validator_name,
                    validator_status=log.validation_result.outcome,
                    property_path=log.property_path,
                    failure_reason=failure_reason,
                    error_spans=error_spans,
                )
            )
        return summaries
