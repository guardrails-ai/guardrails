from typing import Dict, Union, overload

from pydantic import Field

from guardrails.utils.logs_utils import ArbitraryModel, GuardHistory
from guardrails.utils.reask_utils import ReAsk


class ValidationOutcome(ArbitraryModel):
    raw_llm_output: str = Field(
        description="The raw, unchanged output from the LLM call."
    )
    validated_output: Union[str, Dict, ReAsk, None] = Field(
        description="The validated, and potentially fixed,"
        " output from the LLM call after passing through validation."
    )
    validation_passed: bool = Field(
        description="A boolean to indicate whether or not"
        " the LLM output passed validation."
        "  If this is False, the validated_output may be invalid."
    )

    @overload
    def __init__(
        self, raw_llm_output: str, validated_output: str, validation_passed: bool
    ):
        ...

    @overload
    def __init__(
        self, raw_llm_output: str, validated_output: Dict, validation_passed: bool
    ):
        ...

    @overload
    def __init__(
        self, raw_llm_output: str, validated_output: ReAsk, validation_passed: bool
    ):
        ...

    @overload
    def __init__(
        self, raw_llm_output: str, validated_output: None, validation_passed: bool
    ):
        ...

    def __init__(
        self,
        raw_llm_output: str,
        validated_output: Union[str, Dict, ReAsk, None],
        validation_passed: bool,
    ):
        super().__init__(
            raw_llm_output=raw_llm_output,
            validated_output=validated_output,
            validation_passed=validation_passed,
        )
        self.raw_llm_output = raw_llm_output
        self.validated_output = validated_output
        self.validation_passed = validation_passed

    @classmethod
    def from_guard_history(cls, guard_history: GuardHistory):
        raw_output = guard_history.output
        validated_output = guard_history.validated_output
        any_validations_failed = len(guard_history.failed_validations) > 0
        if isinstance(validated_output, str):
            return TextOutcome(
                raw_llm_output=raw_output,
                validated_output=validated_output,
                validation_passed=any_validations_failed,
            )
        else:
            # TODO: Why does instantiation collapse validated_output to a dict?
            return StructuredOutcome(
                raw_llm_output=raw_output,
                validated_output=validated_output,
                validation_passed=any_validations_failed,
            )


class TextOutcome(ValidationOutcome):
    validated_output: Union[str, ReAsk, None] = Field(
        description="The validated, and potentially fixed,"
        " output from the LLM call after passing through validation."
    )

    def __init__(
        self,
        raw_llm_output: str,
        validated_output: Union[str, ReAsk, None],
        validation_passed: bool,
    ):
        super().__init__(raw_llm_output, validated_output, validation_passed)


class StructuredOutcome(ValidationOutcome):
    validated_output: Union[Dict, ReAsk, None] = Field(
        description="The validated, and potentially fixed,"
        " output from the LLM call after passing through validation."
    )

    def __init__(
        self,
        raw_llm_output: str,
        validated_output: Union[Dict, ReAsk, None],
        validation_passed: bool,
    ):
        super().__init__(raw_llm_output, validated_output, validation_passed)
