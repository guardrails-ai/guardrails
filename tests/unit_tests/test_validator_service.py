import pytest
import guardrails.validator_service as vs
from guardrails.datatypes import FieldValidation
from guardrails.utils.logs_utils import FieldValidationLogs
from .mocks import MockAsyncValidatorService


empty_field_validation=FieldValidation(
    key='mock-key',
    value='mock-value',
    validators=[],
    children=[]
)
empty_field_validation_logs=FieldValidationLogs(
    validator_logs=[],
    children={}
)

@pytest.mark.asyncio
async def test_async_validate(mocker):
    mocker.patch(
        "guardrails.validator_service.AsyncValidatorService",
        new=MockAsyncValidatorService
    )
    validated_value, validated_metadata = await vs.async_validate(
        value=True,
        metadata={},
        validator_setup=empty_field_validation,
        validation_logs=empty_field_validation_logs
    )

    assert validated_value is True
    assert validated_metadata == { 'async': True }