import pytest
from guardrails.validator_service import AsyncValidatorService
from guardrails.datatypes import FieldValidation
from guardrails.utils.logs_utils import FieldValidationLogs
from .mocks import MockAsyncValidatorService, MockSequentialValidatorService, MockLoop


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
avs = AsyncValidatorService()

def test_validate_with_running_loop(mocker):
    with pytest.raises(RuntimeError) as e_info:
        mockLoop = MockLoop(True)
        mocker.patch(
            "asyncio.get_event_loop",
            return_value=mockLoop
        )
        avs.validate(
            value=True,
            metadata={},
            validator_setup=empty_field_validation,
            validation_logs=empty_field_validation_logs
        )

        assert str(e_info) == 'Async event loop found, please call `validate_async` instead.'

def test_validate_without_running_loop(mocker):
    mockLoop = MockLoop(False)
    mocker.patch(
        "asyncio.get_event_loop",
        return_value=mockLoop
    )
    async_validate_mock = mocker.MagicMock(return_value=('async_validate_mock', { 'async': True }))
    mocker.patch.object(
        avs,
        'async_validate',
        async_validate_mock
    )
    loop_spy = mocker.spy(mockLoop, 'run_until_complete')

    validated_value, validated_metadata = avs.validate(
        value=True,
        metadata={},
        validator_setup=empty_field_validation,
        validation_logs=empty_field_validation_logs
    )

    assert loop_spy.call_count == 1
    async_validate_mock.assert_called_once_with(
        True,
        {},
        empty_field_validation,
        empty_field_validation_logs
    )
    assert validated_value == 'async_validate_mock'
    assert validated_metadata == { 'async': True }

@pytest.mark.asyncio
async def test_async_validate_with_children(mocker):
    validate_dependents_mock = mocker.patch.object(
        avs,
        'validate_dependents'
    )
    
    run_validators_mock = mocker.patch.object(
        avs,
        'run_validators'
    )
    run_validators_mock.return_value = ('run_validators_mock', { 'async': True })

    field_validation=FieldValidation(
        key='mock-parent-key',
        value='mock-parent-value',
        validators=[],
        children=[empty_field_validation]
    )

    validated_value, validated_metadata = await avs.async_validate(
        value=True,
        metadata={},
        validator_setup=field_validation,
        validation_logs=empty_field_validation_logs
    )

    assert validate_dependents_mock.call_count == 1
    validate_dependents_mock.assert_called_once_with(
        True,
        {},
        field_validation,
        empty_field_validation_logs
    )

    assert run_validators_mock.call_count == 1
    run_validators_mock.assert_called_once_with(
        empty_field_validation_logs,
        field_validation,
        True,
        {}
    )

    assert validated_value == 'run_validators_mock'
    assert validated_metadata == { 'async': True }
