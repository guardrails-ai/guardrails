from datetime import datetime
from typing import Any, Dict, Optional

from guardrails_api_client import (
    ValidatorLog as IValidatorLog,
    ValidatorLogInstanceId,
    ValidatorLogValidationResult,
)
from guardrails.classes.generic.arbitrary_model import ArbitraryModel
from guardrails.classes.validation.validation_result import ValidationResult
from guardrails.utils.casting_utils import to_int


class ValidatorLogs(IValidatorLog, ArbitraryModel):
    """Logs for a single validator."""

    validator_name: str
    registered_name: str
    value_before_validation: Any
    validation_result: Optional[ValidationResult] = None
    value_after_validation: Optional[Any] = None
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    instance_id: Optional[int] = None
    property_path: str

    def to_interface(self) -> IValidatorLog:
        start_time = self.start_time.isoformat() if self.start_time else None
        end_time = self.end_time.isoformat() if self.end_time else None
        return IValidatorLog(
            validator_name=self.validator_name,  # type: ignore - pyright doesn't understand aliases
            registered_name=self.registered_name,  # type: ignore - pyright doesn't understand aliases
            instance_id=ValidatorLogInstanceId(self.instance_id),  # type: ignore - pyright doesn't understand aliases
            property_path=self.property_path,  # type: ignore - pyright doesn't understand aliases
            value_before_validation=self.value_before_validation,  # type: ignore - pyright doesn't understand aliases
            value_after_validation=self.value_after_validation,  # type: ignore - pyright doesn't understand aliases
            validation_result=ValidatorLogValidationResult(self.validation_result),  # type: ignore - pyright doesn't understand aliases
            start_time=start_time,  # type: ignore - pyright doesn't understand aliases
            end_time=end_time,  # type: ignore - pyright doesn't understand aliases
        )

    def to_dict(self) -> Dict[str, Any]:
        return self.to_interface().to_dict()

    @classmethod
    def from_interface(cls, i_validator_log: IValidatorLog) -> "ValidatorLogs":
        instance_id = (
            i_validator_log.instance_id.actual_instance
            if i_validator_log.instance_id
            else None
        )
        validation_result = (
            ValidationResult.from_interface(
                i_validator_log.validation_result.actual_instance
            )
            if (
                i_validator_log.validation_result
                and i_validator_log.validation_result.actual_instance
            )
            else None
        )

        start_time = i_validator_log.start_time
        if isinstance(start_time, str):
            start_time = datetime.fromisoformat(start_time)

        end_time = i_validator_log.end_time
        if isinstance(end_time, str):
            end_time = datetime.fromisoformat(end_time)

        return cls(
            validator_name=i_validator_log.validator_name,
            registered_name=i_validator_log.registered_name,
            instance_id=to_int(instance_id) or 0,
            property_path=i_validator_log.property_path,
            value_before_validation=i_validator_log.value_before_validation,
            value_after_validation=i_validator_log.value_after_validation,
            validation_result=validation_result,
            start_time=start_time,
            end_time=end_time,
        )

    @classmethod
    def from_dict(cls, obj: Dict[str, Any]) -> "ValidatorLogs":
        i_validator_log = IValidatorLog.from_dict(obj)
        return cls.from_interface(i_validator_log)  # type: ignore
