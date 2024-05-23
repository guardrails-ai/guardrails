from datetime import datetime
from typing import Any, Optional

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
