from typing import Any, Dict, List, Optional, Union

from pydantic import Field

from guardrails_api_client import (
    Outputs as IOutputs,
    OutputsParsedOutput,
    OutputsValidationResponse,
)
from guardrails.constants import error_status, fail_status, not_run_status, pass_status
from guardrails.classes.llm.llm_response import LLMResponse
from guardrails.classes.generic.arbitrary_model import ArbitraryModel
from guardrails.classes.validation.validator_logs import ValidatorLogs
from guardrails.actions.reask import ReAsk
from guardrails.classes.validation.validation_result import (
    ErrorSpan,
    FailResult,
    ValidationResult,
)


class Outputs(IOutputs, ArbitraryModel):
    """Outputs represent the data that is output from the validation loop.

    Attributes:
        llm_response_info (Optional[LLMResponse]): Information from the LLM response
        raw_output (Optional[str]): The exact output from the LLM.
        parsed_output (Optional[Union[str, List, Dict]]): The output parsed from the LLM
            response as it was passed into validation.
        validation_response (Optional[Union[str, ReAsk, List, Dict]]): The response
            from the validation process.
        guarded_output (Optional[Union[str, List, Dict]]): Any valid values after
            undergoing validation.
            Some values may be "fixed" values that were corrected during validation.
            This property may be a partial structure if field level reasks occur.
        reasks (List[ReAsk]): Information from the validation process used to construct
            a ReAsk to the LLM on validation failure. Default [].
        validator_logs (List[ValidatorLogs]): The results of each individual
            validation. Default [].
        error (Optional[str]): The error message from any exception that raised
            and interrupted the process.
        exception (Optional[Exception]): The exception that interrupted the process.
    """

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
                and log.validation_result.outcome == "fail"
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
            all_fail_results.extend(reask.fail_results)

        all_reasks_have_fixes = all(
            list(fail.fix_value is not None for fail in all_fail_results)
        )

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

    def to_interface(self) -> IOutputs:
        return IOutputs(
            llm_response_info=(  # type: ignore - pydantic alias
                self.llm_response_info.to_interface()
                if self.llm_response_info
                else None
            ),
            raw_output=self.raw_output,  # type: ignore - pydantic alias
            parsed_output=(  # type: ignore - pydantic alias
                OutputsParsedOutput(self.parsed_output) if self.parsed_output else None
            ),
            validation_response=(  # type: ignore - pydantic alias
                OutputsValidationResponse(self.validation_response)
                if self.validation_response
                else None
            ),
            guarded_output=(  # type: ignore - pydantic alias
                OutputsParsedOutput(self.guarded_output)
                if self.guarded_output
                else None
            ),
            reasks=self.reasks,  # type: ignore - pydantic alias
            validator_logs=[  # type: ignore - pydantic alias
                v.to_interface()
                for v in self.validator_logs
                if isinstance(v, ValidatorLogs)
            ],
            error=self.error,
        )

    def to_dict(self) -> Dict[str, Any]:
        return self.to_interface().to_dict()

    @classmethod
    def from_interface(cls, i_outputs: IOutputs) -> "Outputs":
        reasks = []
        if i_outputs.reasks:
            reasks = [ReAsk.from_interface(r) for r in i_outputs.reasks]

        validator_logs = []
        if i_outputs.validator_logs:
            validator_logs = [
                ValidatorLogs.from_interface(v) for v in i_outputs.validator_logs
            ]

        return cls(
            llm_response_info=(  # type: ignore
                LLMResponse.from_interface(i_outputs.llm_response_info)
                if i_outputs.llm_response_info
                else None
            ),
            raw_output=i_outputs.raw_output,  # type: ignore
            parsed_output=(  # type: ignore
                i_outputs.parsed_output.actual_instance
                if i_outputs.parsed_output
                else None
            ),
            validation_response=(  # type: ignore
                i_outputs.validation_response.actual_instance
                if i_outputs.validation_response
                else None
            ),
            guarded_output=(  # type: ignore
                i_outputs.guarded_output.actual_instance
                if i_outputs.guarded_output
                else None
            ),
            reasks=reasks,  # type: ignore
            validator_logs=validator_logs,  # type: ignore
            error=i_outputs.error,
        )

    @classmethod
    def from_dict(cls, obj: Dict[str, Any]) -> "Outputs":
        i_outputs = IOutputs.from_dict(obj) or IOutputs()

        return cls.from_interface(i_outputs)
