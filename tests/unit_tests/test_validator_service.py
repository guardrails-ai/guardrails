import pytest

import guardrails.validator_service as vs
from guardrails.datatypes import FieldValidation
from guardrails.utils.logs_utils import FieldValidationLogs

from .mocks import MockAsyncValidatorService, MockLoop, MockSequentialValidatorService

empty_field_validation = FieldValidation(
    key="mock-key", value="mock-value", validators=[], children=[]
)
empty_field_validation_logs = FieldValidationLogs(validator_logs=[], children={})


@pytest.mark.asyncio
async def test_async_validate(mocker):
    mocker.patch(
        "guardrails.validator_service.AsyncValidatorService",
        new=MockAsyncValidatorService,
    )
    validated_value, validated_metadata = await vs.async_validate(
        value=True,
        metadata={},
        validator_setup=empty_field_validation,
        validation_logs=empty_field_validation_logs,
    )

    assert validated_value == "MockAsyncValidatorService.async_validate"
    assert validated_metadata == {"async": True}


def test_validate_with_running_loop(mocker):
    mockLoop = MockLoop(True)
    mocker.patch(
        "guardrails.validator_service.AsyncValidatorService",
        new=MockAsyncValidatorService,
    )
    mocker.patch(
        "guardrails.validator_service.SequentialValidatorService",
        new=MockSequentialValidatorService,
    )
    mocker.patch("asyncio.get_event_loop", return_value=mockLoop)

    validated_value, validated_metadata = vs.validate(
        value=True,
        metadata={},
        validator_setup=empty_field_validation,
        validation_logs=empty_field_validation_logs,
    )

    assert validated_value == "MockSequentialValidatorService.validate"
    assert validated_metadata == {"sync": True}


def test_validate_without_running_loop(mocker):
    mockLoop = MockLoop(False)
    mocker.patch(
        "guardrails.validator_service.AsyncValidatorService",
        new=MockAsyncValidatorService,
    )
    mocker.patch(
        "guardrails.validator_service.SequentialValidatorService",
        new=MockSequentialValidatorService,
    )
    mocker.patch("asyncio.get_event_loop", return_value=mockLoop)
    validated_value, validated_metadata = vs.validate(
        value=True,
        metadata={},
        validator_setup=empty_field_validation,
        validation_logs=empty_field_validation_logs,
    )

    assert validated_value == "MockAsyncValidatorService.validate"
    assert validated_metadata == {"sync": True}


def test_validate_loop_runtime_error(mocker):
    mocker.patch(
        "guardrails.validator_service.AsyncValidatorService",
        new=MockAsyncValidatorService,
    )
    mocker.patch(
        "guardrails.validator_service.SequentialValidatorService",
        new=MockSequentialValidatorService,
    )
    # raise RuntimeError in `get_event_loop`
    mocker.patch("asyncio.get_event_loop", side_effect=RuntimeError)

    validated_value, validated_metadata = vs.validate(
        value=True,
        metadata={},
        validator_setup=empty_field_validation,
        validation_logs=empty_field_validation_logs,
    )

    assert validated_value == "MockSequentialValidatorService.validate"
    assert validated_metadata == {"sync": True}
