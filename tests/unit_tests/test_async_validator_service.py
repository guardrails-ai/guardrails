import asyncio

import pytest

from guardrails.classes.history.iteration import Iteration
from guardrails.datatypes import FieldValidation
from guardrails.utils.logs_utils import ValidatorLogs
from guardrails.validator_service import AsyncValidatorService
from guardrails.validators import PassResult

from .mocks import MockLoop
from .mocks.mock_validator import create_mock_validator

empty_field_validation = FieldValidation(
    key="mock-key", value="mock-value", validators=[], children=[]
)
avs = AsyncValidatorService()


def test_validate_with_running_loop(mocker):
    iteration = Iteration()
    with pytest.raises(RuntimeError) as e_info:
        mock_loop = MockLoop(True)
        mocker.patch("asyncio.get_event_loop", return_value=mock_loop)
        avs.validate(
            value=True,
            metadata={},
            validator_setup=empty_field_validation,
            iteration=iteration,
        )

        assert (
            str(e_info)
            == "Async event loop found, please call `validate_async` instead."
        )


def test_validate_without_running_loop(mocker):
    mock_loop = MockLoop(False)
    mocker.patch("asyncio.get_event_loop", return_value=mock_loop)
    async_validate_mock = mocker.MagicMock(
        return_value=("async_validate_mock", {"async": True})
    )
    mocker.patch.object(avs, "async_validate", async_validate_mock)
    loop_spy = mocker.spy(mock_loop, "run_until_complete")

    iteration = Iteration()

    validated_value, validated_metadata = avs.validate(
        value=True,
        metadata={},
        validator_setup=empty_field_validation,
        iteration=iteration,
    )

    assert loop_spy.call_count == 1
    async_validate_mock.assert_called_once_with(
        True, {}, empty_field_validation, iteration
    )
    assert validated_value == "async_validate_mock"
    assert validated_metadata == {"async": True}


@pytest.mark.asyncio
async def test_async_validate_with_children(mocker):
    validate_dependents_mock = mocker.patch.object(avs, "validate_dependents")

    run_validators_mock = mocker.patch.object(avs, "run_validators")
    run_validators_mock.return_value = ("run_validators_mock", {"async": True})

    field_validation = FieldValidation(
        key="mock-parent-key",
        value="mock-parent-value",
        validators=[],
        children=[empty_field_validation],
    )
    iteration = Iteration()

    validated_value, validated_metadata = await avs.async_validate(
        value=True,
        metadata={},
        validator_setup=field_validation,
        iteration=iteration,
    )

    assert validate_dependents_mock.call_count == 1
    validate_dependents_mock.assert_called_once_with(
        True, {}, field_validation, iteration
    )

    assert run_validators_mock.call_count == 1
    run_validators_mock.assert_called_once_with(iteration, field_validation, True, {})

    assert validated_value == "run_validators_mock"
    assert validated_metadata == {"async": True}


@pytest.mark.asyncio
async def test_async_validate_without_children(mocker):
    validate_dependents_mock = mocker.patch.object(avs, "validate_dependents")

    run_validators_mock = mocker.patch.object(avs, "run_validators")
    run_validators_mock.return_value = ("run_validators_mock", {"async": True})

    iteration = Iteration()

    validated_value, validated_metadata = await avs.async_validate(
        value=True,
        metadata={},
        validator_setup=empty_field_validation,
        iteration=iteration,
    )

    assert validate_dependents_mock.call_count == 0

    assert run_validators_mock.call_count == 1
    run_validators_mock.assert_called_once_with(
        iteration, empty_field_validation, True, {}
    )

    assert validated_value == "run_validators_mock"
    assert validated_metadata == {"async": True}


@pytest.mark.asyncio
async def test_validate_dependents(mocker):
    async def mock_async_validate(v, md, *args):
        return (f"new-{v}", md)

    async_validate_mock = mocker.patch.object(
        avs, "async_validate", side_effect=mock_async_validate
    )

    gather_spy = mocker.spy(asyncio, "gather")

    child_one = FieldValidation(
        key="child-one-key", value="child-one-value", validators=[], children=[]
    )
    child_two = FieldValidation(
        key="child-two-key", value="child-two-value", validators=[], children=[]
    )
    field_validation = FieldValidation(
        key="mock-parent-key",
        value={"child-one-key": "child-one-value", "child-two-key": "child-two-value"},
        validators=[],
        children=[child_one, child_two],
    )
    iteration = Iteration()

    validated_value, validated_metadata = await avs.validate_dependents(
        value=field_validation.value,
        metadata={},
        validator_setup=field_validation,
        iteration=iteration,
    )

    assert gather_spy.call_count == 1

    assert async_validate_mock.call_count == 2
    async_validate_mock.assert_any_call(child_one.value, {}, child_one, iteration)
    async_validate_mock.assert_any_call(child_two.value, {}, child_two, iteration)

    assert validated_value == {
        "child-one-key": "new-child-one-value",
        "child-two-key": "new-child-two-value",
    }
    assert validated_metadata == {}


@pytest.mark.asyncio
async def test_run_validators(mocker):
    group_validators_mock = mocker.patch.object(avs, "group_validators")
    fix_validator_type = create_mock_validator("fix_validator", "fix")
    fix_validator = fix_validator_type()
    noop_validator_type = create_mock_validator("noop_validator")
    noop_validator_1 = noop_validator_type()
    noop_validator_type = create_mock_validator("noop_validator")
    noop_validator_2 = noop_validator_type()
    noop_validator_2.run_in_separate_process = True
    group_validators_mock.return_value = [
        ("fix", [fix_validator]),
        ("noop", [noop_validator_1, noop_validator_2]),
    ]

    def mock_run_validator(iteration, validator, value, metadata):
        return ValidatorLogs(
            validator_name=validator.name,
            value_before_validation=value,
            validation_result=PassResult(),
        )

    run_validator_mock = mocker.patch.object(
        avs, "run_validator", side_effect=mock_run_validator
    )

    mock_loop = MockLoop(True)
    run_in_executor_spy = mocker.spy(mock_loop, "run_in_executor")
    get_running_loop_mock = mocker.patch(
        "asyncio.get_running_loop", return_value=mock_loop
    )

    async def mock_gather(*args):
        return args

    asyancio_gather_mock = mocker.patch("asyncio.gather", side_effect=mock_gather)

    iteration = Iteration()

    value, metadata = await avs.run_validators(
        value=empty_field_validation.value,
        metadata={},
        validator_setup=empty_field_validation,
        iteration=iteration,
    )

    assert get_running_loop_mock.call_count == 1

    assert group_validators_mock.call_count == 1
    group_validators_mock.assert_called_once_with(empty_field_validation.validators)

    assert run_in_executor_spy.call_count == 1
    run_in_executor_spy.assert_called_once_with(
        avs.multiprocessing_executor,
        run_validator_mock,
        iteration,
        noop_validator_2,
        empty_field_validation.value,
        {},
    )

    assert run_validator_mock.call_count == 3

    assert asyancio_gather_mock.call_count == 1

    assert value == empty_field_validation.value
    assert metadata == {}


@pytest.mark.asyncio
async def test_run_validators_with_override(mocker):
    group_validators_mock = mocker.patch.object(avs, "group_validators")
    override_validator_type = create_mock_validator("override")
    override_validator = override_validator_type()
    override_validator.override_value_on_pass = True

    group_validators_mock.return_value = [("exception", [override_validator])]

    run_validator_mock = mocker.patch.object(avs, "run_validator")
    run_validator_mock.return_value = ValidatorLogs(
        validator_name="override",
        value_before_validation="mock-value",
        validation_result=PassResult(value_override="override"),
    )

    mock_loop = MockLoop(True)
    run_in_executor_spy = mocker.spy(mock_loop, "run_in_executor")
    get_running_loop_mock = mocker.patch(
        "asyncio.get_running_loop", return_value=mock_loop
    )

    asyancio_gather_mock = mocker.patch("asyncio.gather")

    iteration = Iteration()

    value, metadata = await avs.run_validators(
        value=empty_field_validation.value,
        metadata={},
        validator_setup=empty_field_validation,
        iteration=iteration,
    )

    assert get_running_loop_mock.call_count == 1

    assert group_validators_mock.call_count == 1
    group_validators_mock.assert_called_once_with(empty_field_validation.validators)

    assert run_in_executor_spy.call_count == 0

    assert run_validator_mock.call_count == 1

    assert asyancio_gather_mock.call_count == 0

    assert value == "override"
    assert metadata == {}


# TODO
@pytest.mark.asyncio
async def test_run_validators_with_failures(mocker):
    assert True is True
