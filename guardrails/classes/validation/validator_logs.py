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
    """Logs for a single validator execution.

    Attributes:
        validator_name (str): The class name of the validator
        registered_name (str): The snake_cased id of the validator
        property_path (str): The JSON path to the property being validated
        value_before_validation (Any): The value before validation
        value_after_validation (Optional[Any]): The value after validation;
            could be different if `value_override`s or `fix`es are applied
        validation_result (Optional[ValidationResult]): The result of the validation
        start_time (Optional[datetime]): The time the validation started
        end_time (Optional[datetime]): The time the validation ended
        instance_id (Optional[int]): The unique id of this instance of the validator
    """

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
        validation_result = (
            ValidatorLogValidationResult(self.validation_result)
            if self.validation_result
            else None
        )
        # pyright doesn't understand aliases so all type aliases are ignored.
        return IValidatorLog(
            validator_name=self.validator_name,  # type: ignore
            registered_name=self.registered_name,  # type: ignore
            instance_id=ValidatorLogInstanceId(self.instance_id),  # type: ignore
            property_path=self.property_path,  # type: ignore
            value_before_validation=self.value_before_validation,  # type: ignore
            value_after_validation=self.value_after_validation,  # type: ignore
            validation_result=validation_result,  # type: ignore
            start_time=start_time,  # type: ignore
            end_time=end_time,  # type: ignore
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
