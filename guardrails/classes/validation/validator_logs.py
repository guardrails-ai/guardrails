from datetime import datetime
from typing import Any, Dict, Optional
from typing_extensions import deprecated
from pydantic import Field, field_validator, field_serializer
from guardrails_ai.types import ValidationResult, Outcome, PassResult, FailResult
from guardrails.classes.generic.arbitrary_model import ArbitraryModel


class ValidatorLogs(ArbitraryModel):
    """Logs for a single validator execution."""

    validator_name: str = Field(
        description="The class name of the validator.", alias="validatorName"
    )
    registered_name: str = Field(
        description="The registry id of the validator.", alias="registeredName"
    )
    value_before_validation: Any = Field(alias="valueBeforeValidation")
    validation_result: Optional[ValidationResult] = Field(
        default=None, alias="validationResult"
    )
    value_after_validation: Optional[Any] = Field(
        default=None, alias="valueAfterValidation"
    )
    start_time: Optional[datetime] = Field(default=None, alias="startTime")
    end_time: Optional[datetime] = Field(default=None, alias="endTime")
    instance_id: Optional[int] = Field(default=None, alias="instanceId")
    property_path: str = Field(
        description="The JSON path to the property which was validated that produced"
        " this log.",
        alias="propertyPath",
    )

    @field_serializer("start_time")
    def serialize_start_time(self, start_time: datetime | None) -> str | None:
        if start_time is None:
            return None

        return start_time.isoformat()

    @field_serializer("end_time")
    def serialize_end_time(self, end_time: datetime | None) -> str | None:
        if end_time is None:
            return None

        return end_time.isoformat()

    # NOTE: It shouldn't be necessary to add this serializer just to call model_dump,
    #   but it is to get the correct serialized output.
    @field_serializer("validation_result")
    def serialize_validation_result(
        self, validation_result: ValidationResult | None
    ) -> dict[str, Any] | None:
        if validation_result is None:
            return None
        return validation_result.model_dump(exclude_none=True, by_alias=True)

    @field_validator("validation_result", mode="before")
    @classmethod
    def deserialize_validation_result(
        cls, validation_result: Any
    ) -> ValidationResult | None:
        if validation_result is None:
            return None
        elif isinstance(validation_result, ValidationResult):
            return validation_result
        elif isinstance(validation_result, dict):
            outcome = validation_result.get("outcome")
            if outcome == Outcome.PASS:
                return PassResult.model_validate(validation_result)
            elif outcome == Outcome.FAIL:
                return FailResult.model_validate(validation_result)
        return ValidationResult.model_validate(validation_result)

    @deprecated("Use ValidatorLogs.model_dump() instead.")
    def to_interface(self) -> dict[str, Any]:
        return self.model_dump(exclude_none=True, by_alias=True)

    @deprecated("Use ValidatorLogs.model_dump() instead.")
    def to_dict(self) -> Dict[str, Any]:
        return self.model_dump(exclude_none=True, by_alias=True)

    @classmethod
    @deprecated("Use ValidatorLogs.model_validate() instead.")
    def from_interface(cls, i_validator_log: Any) -> "ValidatorLogs":
        return cls.model_validate(i_validator_log)

    @classmethod
    @deprecated("Use ValidatorLogs.model_validate() instead.")
    def from_dict(cls, obj: Any) -> "ValidatorLogs":
        return cls.model_validate(obj)
