from __future__ import annotations
from typing import Any, Dict, List, Optional, Union
from typing_extensions import deprecated

from pydantic import Field, field_serializer, field_validator, ValidationError

from guardrails_ai.types import FailResult, ValidationResult, ErrorSpan, Outcome

from guardrails.constants import error_status, fail_status, not_run_status, pass_status
from guardrails.classes.llm.llm_response import LLMResponse
from guardrails.classes.generic.arbitrary_model import ArbitraryModel
from guardrails.classes.validation.validator_logs import ValidatorLogs
from guardrails.actions.reask import to_reask, ReAsk


class Outputs(ArbitraryModel):
    """Outputs represent the data that is output from the validation loop."""

    llm_response_info: Optional[LLMResponse] = Field(
        description="Information from the LLM response.", default=None
    )
    raw_output: Optional[str] = Field(
        description="The exact output from the LLM.", default=None
    )
    parsed_output: Optional[Union[str, List, Dict]] = Field(
        description="The output parsed from the LLM response"
        "as it was passed into validation.",
        default=None,
    )
    validation_response: Optional[Union[str, ReAsk, List, Dict]] = Field(
        description="The response from the validation process.", default=None
    )
    guarded_output: Optional[Union[str, List, Dict]] = Field(
        description="""Any valid values after undergoing validation.

        Some values may be "fixed" values that were corrected during validation.
        This property may be a partial structure if field level reasks occur.""",
        default=None,
    )
    reasks: List[ReAsk] = Field(
        description="Information from the validation process"
        "used to construct a ReAsk to the LLM on validation failure.",
        default_factory=list,
    )
    # TODO: Rename this;
    validator_logs: List[ValidatorLogs] = Field(
        description="The results of each individual validation.", default_factory=list
    )
    error: Optional[str] = Field(
        description="The error message from any exception"
        "that raised and interrupted the process.",
        default=None,
    )
    exception: Optional[Exception] = Field(
        description="The exception that interrupted the process.", default=None
    )

    @field_validator("validation_response", mode="before")
    @classmethod
    def deserialize_validation_response(
        cls, validation_response: Any | None
    ) -> str | ReAsk | List | Dict | None:
        if isinstance(validation_response, ReAsk):
            return validation_response
        if validation_response and isinstance(validation_response, dict):
            try:
                return to_reask(validation_response)
            except ValidationError:
                return validation_response
        return validation_response

    @field_validator("reasks", mode="before")
    @classmethod
    def deserialize_reasks(cls, reasks: Any) -> List[ReAsk]:
        if reasks and isinstance(reasks, list):
            return [to_reask(r) if not isinstance(r, ReAsk) else r for r in reasks]
        return []

    @field_serializer("exception")
    def serialize_exception(self, exception: Exception | None) -> str | None:
        if exception:
            return str(exception)
        return None

    @field_validator("exception", mode="before")
    @classmethod
    def deserialize_exception(cls, exception: Any) -> Exception | None:
        if isinstance(exception, Exception):
            return exception
        if exception and isinstance(exception, str):
            return Exception(exception)
        return None

    def _all_empty(self) -> bool:
        return (
            self.llm_response_info is None
            and self.parsed_output is None
            and self.validation_response is None
            and self.guarded_output is None
            and len(self.reasks) == 0
            and len(self.validator_logs) == 0
            and self.error is None
        )

    @property
    def failed_validations(self) -> List[ValidatorLogs]:
        """Returns the validator logs for any validation that failed."""
        return list(
            [
                log
                for log in self.validator_logs
                if log.validation_result is not None
                and isinstance(log.validation_result, ValidationResult)
                and log.validation_result.outcome == Outcome.FAIL
            ]
        )

    @property
    def error_spans_in_output(self) -> List[ErrorSpan]:
        """The error spans from the LLM response.

        These indices are relative to the complete LLM output.
        """
        # map of total length to validator
        total_len_by_validator = {}
        spans_in_output = []
        for log in self.validator_logs:
            validator_name = log.validator_name
            if total_len_by_validator.get(validator_name) is None:
                total_len_by_validator[validator_name] = 0
            result = log.validation_result
            if isinstance(result, FailResult):
                if result.error_spans is not None:
                    for error_span in result.error_spans:
                        spans_in_output.append(
                            ErrorSpan(
                                start=error_span.start
                                + total_len_by_validator[validator_name],
                                end=error_span.end
                                + total_len_by_validator[validator_name],
                                reason=error_span.reason,
                            )
                        )
            if isinstance(result, ValidationResult):
                if result and result.validated_chunk is not None:
                    total_len_by_validator[validator_name] += len(
                        result.validated_chunk
                    )
        return spans_in_output

    @property
    def status(self) -> str:
        """Representation of the end state of the validation run.

        OneOf: pass, fail, error, not run
        """
        all_fail_results: List[FailResult] = []
        for reask in self.reasks:
            all_fail_results.extend(reask.fail_results or [])

        print("all_fail_results: ", all_fail_results)

        all_reasks_have_fixes = all(
            list(fail.fix_value is not None for fail in all_fail_results)
        )

        print("all_reasks_have_fixes: ", all_reasks_have_fixes)

        if self._all_empty() is True:
            return not_run_status
        elif self.error:
            return error_status
        elif not all_reasks_have_fixes:
            return fail_status
        elif self.guarded_output is None and isinstance(
            self.validation_response, ReAsk
        ):
            return fail_status
        return pass_status

    @deprecated("Use Outputs.model_dump() instead.")
    def to_interface(self) -> dict[str, Any]:
        return self.model_dump(exclude_none=True, by_alias=True)

    @deprecated("Use Outputs.model_dump() instead.")
    def to_dict(self) -> Dict[str, Any]:
        return self.model_dump(exclude_none=True, by_alias=True)

    @classmethod
    @deprecated("Use Outputs.model_validate() instead.")
    def from_interface(cls, i_outputs: Any) -> "Outputs":
        return cls.model_validate(i_outputs)

    @classmethod
    @deprecated("Use Outputs.model_validate() instead.")
    def from_dict(cls, obj: Any) -> "Outputs":
        return cls.model_validate(obj)
