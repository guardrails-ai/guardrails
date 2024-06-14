from datetime import datetime
from typing import Any, Dict, Optional

from guardrails_api_client import ValidatorLog
from guardrails.classes.generic.arbitrary_model import ArbitraryModel
from guardrails.classes.validation.validation_result import ValidationResult


class ValidatorLogs(ValidatorLog, ArbitraryModel):
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

    def to_dict(self) -> Dict[str, Any]:
        i_validator_logs = ValidatorLog(
            validator_name=self.validator_name,  # type: ignore
            registered_name=self.registered_name,  # type: ignore
            value_before_validation=self.value_before_validation,  # type: ignore
            value_after_validation=self.value_after_validation,  # type: ignore
            property_path=self.property_path,  # type: ignore
        )

        i_validator_log = i_validator_logs.model_dump(
            by_alias=True,
            exclude_none=True,
        )
        if self.instance_id:
            i_validator_log["instanceId"] = self.instance_id
        if self.validation_result:
            i_validator_log["validation_result"] = self.validation_result.to_dict()
        if self.start_time:
            i_validator_log["start_time"] = self.start_time.isoformat()
        if self.end_time:
            i_validator_log["end_time"] = self.end_time.isoformat()

        return i_validator_log
