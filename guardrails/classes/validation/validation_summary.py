# TODO Temp to update once generated class is in
from typing import Iterator, List

from guardrails.classes.generic.arbitrary_model import ArbitraryModel
from guardrails.classes.validation.validation_result import FailResult
from guardrails.classes.validation.validator_logs import ValidatorLogs
from guardrails_api_client import ValidationSummary as IValidationSummary


class ValidationSummary(IValidationSummary, ArbitraryModel):
    @staticmethod
    def _generate_summaries_from_validator_logs(
        validator_logs: List[ValidatorLogs],
    ) -> Iterator["ValidationSummary"]:
        """
        Generate a list of ValidationSummary objects from a list of
        ValidatorLogs objects. Using an iterator to allow serializing
        the summaries to other formats.
        """
        for log in validator_logs:
            validation_result = log.validation_result
            is_fail_result = isinstance(validation_result, FailResult)
            failure_reason = validation_result.error_message if is_fail_result else None
            error_spans = validation_result.error_spans if is_fail_result else []
            yield ValidationSummary(
                validatorName=log.validator_name,
                validatorStatus=log.validation_result.outcome,  # type: ignore
                propertyPath=log.property_path,
                failureReason=failure_reason,
                errorSpans=error_spans,  # type: ignore
            )

    @staticmethod
    def from_validator_logs(
        validator_logs: List[ValidatorLogs],
    ) -> List["ValidationSummary"]:
        summaries = []
        for summary in ValidationSummary._generate_summaries_from_validator_logs(
            validator_logs
        ):
            summaries.append(summary)
        return summaries

    @staticmethod
    def from_validator_logs_only_fails(
        validator_logs: List[ValidatorLogs],
    ) -> List["ValidationSummary"]:
        summaries = []
        for summary in ValidationSummary._generate_summaries_from_validator_logs(
            validator_logs
        ):
            if summary.failure_reason:
                summaries.append(summary)
        return summaries
