from typing import Dict, List, Optional, Sequence, Union

from pydantic import Field

from guardrails.constants import error_status, fail_status, not_run_status, pass_status
from guardrails.utils.llm_response import LLMResponse
from guardrails.utils.logs_utils import ValidatorLogs
from guardrails.utils.pydantic_utils import ArbitraryModel
from guardrails.utils.reask_utils import ReAsk
from guardrails.validator_base import FailResult


class Outputs(ArbitraryModel):
    llm_response_info: Optional[LLMResponse] = Field(
        description="Information from the LLM response.", default=None
    )
    raw_output: Optional[str] = Field(
        description="The exact output from the LLM.", default=None
    )
    parsed_output: Optional[Union[str, Dict]] = Field(
        description="The output parsed from the LLM response"
        "as it was passed into validation.",
        default=None,
    )
    validation_output: Optional[Union[str, ReAsk, Dict]] = Field(
        description="The output from the validation process.", default=None
    )
    validated_output: Optional[Union[str, Dict]] = Field(
        description="The valid output after validation."
        "Could be only a partial structure if field level reasks occur."
        "Could contain fixed values.",
        default=None,
    )
    reasks: Sequence[ReAsk] = Field(
        description="Information from the validation process"
        "used to construct a ReAsk to the LLM on validation failure.",
        default_factory=list,
    )
    # TODO: Rename this;
    # TODO: Add json_path to ValidatorLogs to specify what property it applies to
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
            and self.validation_output is None
            and self.validated_output is None
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
                and log.validation_result.outcome == "fail"
            ]
        )

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
        elif self.validated_output is None and isinstance(
            self.validation_output, ReAsk
        ):
            return fail_status
        return pass_status
