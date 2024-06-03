import asyncio

import pytest

from guardrails.classes.history.iteration import Iteration
from guardrails.classes.validation.validator_logs import ValidatorLogs
from guardrails.validator_base import OnFailAction
from guardrails.validator_service import AsyncValidatorService
from guardrails.validators import PassResult

from .mocks import MockLoop
from .mocks.mock_validator import create_mock_validator

avs = AsyncValidatorService()


def test_validate_with_running_loop(mocker):
    iteration = Iteration()
    with pytest.raises(RuntimeError) as e_info:
        mock_loop = MockLoop(True)
        mocker.patch("asyncio.get_event_loop", return_value=mock_loop)
        avs.validate(
            value=True,
            metadata={},
            validator_map={},
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
        validator_map={},
        iteration=iteration,
    )

    assert loop_spy.call_count == 1
    async_validate_mock.assert_called_once_with(True, {}, {}, iteration, "$", "$")
    assert validated_value == "async_validate_mock"
    assert validated_metadata == {"async": True}


@pytest.mark.asyncio
async def test_async_validate_with_children(mocker):
    validate_children_mock = mocker.patch.object(avs, "validate_children")

    run_validators_mock = mocker.patch.object(avs, "run_validators")
    run_validators_mock.return_value = ("run_validators_mock", {"async": True})

    value = {"a": 1}

    iteration = Iteration()

    validated_value, validated_metadata = await avs.async_validate(
        value=value,
        metadata={},
        validator_map={},
        iteration=iteration,
    )

    assert validate_children_mock.call_count == 1
    validate_children_mock.assert_called_once_with(value, {}, {}, iteration, "$", "$")

    assert run_validators_mock.call_count == 1
    run_validators_mock.assert_called_once_with(iteration, {}, value, {}, "$", "$")

    assert validated_value == "run_validators_mock"
    assert validated_metadata == {"async": True}


@pytest.mark.asyncio
async def test_async_validate_without_children(mocker):
    validate_children_mock = mocker.patch.object(avs, "validate_children")

    run_validators_mock = mocker.patch.object(avs, "run_validators")
    run_validators_mock.return_value = ("run_validators_mock", {"async": True})

    iteration = Iteration()

    validated_value, validated_metadata = await avs.async_validate(
        value="Hello world!",
        metadata={},
        validator_map={},
        iteration=iteration,
    )

    assert validate_children_mock.call_count == 0

    assert run_validators_mock.call_count == 1
    run_validators_mock.assert_called_once_with(
        iteration, {}, "Hello world!", {}, "$", "$"
    )

    assert validated_value == "run_validators_mock"
    assert validated_metadata == {"async": True}


@pytest.mark.asyncio
async def test_validate_children(mocker):
    async def mock_async_validate(v, md, *args):
        return (f"new-{v}", md)

    async_validate_mock = mocker.patch.object(
        avs, "async_validate", side_effect=mock_async_validate
    )

    gather_spy = mocker.spy(asyncio, "gather")

    validator_map = {
        "$.mock-parent-key": [],
        "$.mock-parent-key.child-one-key": [],
        "$.mock-parent-key.child-two-key": [],
    }

    value = {
        "mock-parent-key": {
            "child-one-key": "child-one-value",
            "child-two-key": "child-two-value",
        }
    }

    iteration = Iteration()

    validated_value, validated_metadata = await avs.validate_children(
        value=value.get("mock-parent-key"),
        metadata={},
        validator_map=validator_map,
        iteration=iteration,
        abs_parent_path="$.mock-parent-key",
        ref_parent_path="$.mock-parent-key",
    )

    assert gather_spy.call_count == 1

    assert async_validate_mock.call_count == 2
    async_validate_mock.assert_any_call(
        "child-one-value",
        {},
        validator_map,
        iteration,
        "$.mock-parent-key.child-one-key",
        "$.mock-parent-key.child-one-key",
    )
    async_validate_mock.assert_any_call(
        "child-two-value",
        {},
        validator_map,
        iteration,
        "$.mock-parent-key.child-two-key",
        "$.mock-parent-key.child-two-key",
    )

    assert validated_value == {
        "child-one-key": "new-child-one-value",
        "child-two-key": "new-child-two-value",
    }
    assert validated_metadata == {}


@pytest.mark.asyncio
async def test_run_validators(mocker):
    group_validators_mock = mocker.patch.object(avs, "group_validators")
    fix_validator_type = create_mock_validator("fix_validator", OnFailAction.FIX)
    fix_validator = fix_validator_type()
    noop_validator_type = create_mock_validator("noop_validator")
    noop_validator_1 = noop_validator_type()
    noop_validator_type = create_mock_validator("noop_validator")
    noop_validator_2 = noop_validator_type()
    noop_validator_2.run_in_separate_process = True
    group_validators_mock.return_value = [
        (OnFailAction.FIX, [fix_validator]),
        (OnFailAction.NOOP, [noop_validator_1, noop_validator_2]),
    ]

    def mock_run_validator(iteration, validator, value, metadata, property_path):
        return ValidatorLogs(
            registered_name=validator.name,
            validator_name=validator.name,
            value_before_validation=value,
            validation_result=PassResult(),
            property_path=property_path,
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
        iteration=iteration,
        validator_map={},
        value=True,
        metadata={},
        absolute_property_path="$",
        reference_property_path="$",
    )

    assert get_running_loop_mock.call_count == 1

    assert group_validators_mock.call_count == 1
    group_validators_mock.assert_called_once_with([])

    assert run_in_executor_spy.call_count == 1
    run_in_executor_spy.assert_called_once_with(
        avs.multiprocessing_executor,
        run_validator_mock,
        iteration,
        noop_validator_2,
        True,
        {},
        "$",
    )

    assert run_validator_mock.call_count == 3

    assert asyancio_gather_mock.call_count == 1

    assert value is True
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
        registered_name="override",
        validator_name="override",
        value_before_validation="mock-value",
        validation_result=PassResult(value_override="override"),
        property_path="$",
    )

    mock_loop = MockLoop(True)
    run_in_executor_spy = mocker.spy(mock_loop, "run_in_executor")
    get_running_loop_mock = mocker.patch(
        "asyncio.get_running_loop", return_value=mock_loop
    )

    asyancio_gather_mock = mocker.patch("asyncio.gather")

    iteration = Iteration()

    value, metadata = await avs.run_validators(
        iteration=iteration,
        validator_map={},
        value=True,
        metadata={},
        absolute_property_path="$",
        reference_property_path="$",
    )

    assert get_running_loop_mock.call_count == 1

    assert group_validators_mock.call_count == 1
    group_validators_mock.assert_called_once_with([])

    assert run_in_executor_spy.call_count == 0

    assert run_validator_mock.call_count == 1

    assert asyancio_gather_mock.call_count == 0

    assert value == "override"
    assert metadata == {}


# TODO
@pytest.mark.asyncio
async def test_run_validators_with_failures(mocker):
    assert True is True
