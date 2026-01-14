from datetime import datetime
from unittest.mock import MagicMock, call

from guardrails.actions.filter import Filter
from guardrails.validator_service.validator_service_base import ValidatorRun
import pytest

from guardrails.classes.history.iteration import Iteration
from guardrails.classes.validation.validator_logs import ValidatorLogs
from guardrails.validator_base import OnFailAction, Validator
from guardrails.validator_service.async_validator_service import AsyncValidatorService
from guardrails.classes.validation.validation_result import FailResult, PassResult


avs = AsyncValidatorService()


def test_validate(mocker):
    mock_loop = mocker.MagicMock()
    mock_loop.run_until_complete = mocker.MagicMock(return_value=(True, {}))
    # loop_spy = mocker.spy(mock_loop, "run_until_complete", return_value=(True, {}))
    async_validate_mock = mocker.patch.object(avs, "async_validate")

    iteration = Iteration(
        call_id="mock-call",
        index=0,
    )

    avs.validate(
        value=True,
        metadata={},
        validator_map={},
        iteration=iteration,
        absolute_path="$",
        reference_path="$",
        loop=mock_loop,
    )

    assert mock_loop.run_until_complete.call_count == 1
    async_validate_mock.assert_called_once_with(True, {}, {}, iteration, "$", "$", stream=False)


class TestAsyncValidate:
    @pytest.mark.asyncio
    async def test_with_dictionary(self, mocker):
        validate_children_mock = mocker.patch.object(avs, "validate_children")

        run_validators_mock = mocker.patch.object(
            avs, "run_validators", return_value=("run_validators_mock", {"async": True})
        )

        value = {"a": 1}

        iteration = Iteration(
            call_id="mock-call",
            index=0,
        )

        validated_value, validated_metadata = await avs.async_validate(
            value=value,
            metadata={},
            validator_map={},
            iteration=iteration,
            absolute_path="$",
            reference_path="$",
        )

        assert validate_children_mock.call_count == 1
        validate_children_mock.assert_called_once_with(
            value, {}, {}, iteration, "$", "$", stream=False
        )

        assert run_validators_mock.call_count == 1
        run_validators_mock.assert_called_once_with(
            iteration, {}, value, {}, "$", "$", stream=False
        )

        assert validated_value == "run_validators_mock"
        assert validated_metadata == {"async": True}

    @pytest.mark.asyncio
    async def test_with_list(self, mocker):
        validate_children_mock = mocker.patch.object(avs, "validate_children")

        run_validators_mock = mocker.patch.object(
            avs, "run_validators", return_value=("run_validators_mock", {"async": True})
        )

        value = ["a"]

        iteration = Iteration(
            call_id="mock-call",
            index=0,
        )

        validated_value, validated_metadata = await avs.async_validate(
            value=value,
            metadata={},
            validator_map={},
            iteration=iteration,
            absolute_path="$",
            reference_path="$",
        )

        assert validate_children_mock.call_count == 1
        validate_children_mock.assert_called_once_with(
            value, {}, {}, iteration, "$", "$", stream=False
        )

        assert run_validators_mock.call_count == 1
        run_validators_mock.assert_called_once_with(
            iteration, {}, value, {}, "$", "$", stream=False
        )

        assert validated_value == "run_validators_mock"
        assert validated_metadata == {"async": True}

    @pytest.mark.asyncio
    async def test_without_children(self, mocker):
        validate_children_mock = mocker.patch.object(avs, "validate_children")

        run_validators_mock = mocker.patch.object(avs, "run_validators")
        run_validators_mock.return_value = ("run_validators_mock", {"async": True})

        iteration = Iteration(
            call_id="mock-call",
            index=0,
        )

        validated_value, validated_metadata = await avs.async_validate(
            value="Hello world!",
            metadata={},
            validator_map={},
            iteration=iteration,
            absolute_path="$",
            reference_path="$",
        )

        assert validate_children_mock.call_count == 0

        assert run_validators_mock.call_count == 1
        run_validators_mock.assert_called_once_with(
            iteration, {}, "Hello world!", {}, "$", "$", stream=False
        )

        assert validated_value == "run_validators_mock"
        assert validated_metadata == {"async": True}


class TestValidateChildren:
    @pytest.mark.asyncio
    async def test_with_list(self, mocker):
        mock_async_validate = mocker.patch.object(
            avs,
            "async_validate",
            side_effect=[
                (
                    "mock-child-1-value",
                    {
                        "mock-child-1-metadata": "child-1-metadata",
                        "mock-shared-metadata": "shared-metadata-1",
                    },
                ),
                (
                    "mock-child-2-value",
                    {
                        "mock-child-2-metadata": "child-2-metadata",
                        "mock-shared-metadata": "shared-metadata-2",
                    },
                ),
            ],
        )

        iteration = Iteration(
            call_id="mock-call",
            index=0,
        )

        validator_map = ({"$.*": [MagicMock(spec=Validator)]},)
        value, metadata = await avs.validate_children(
            value=["mock-child-1", "mock-child-2"],
            metadata={"mock-shared-metadata": "shared-metadata"},
            validator_map=validator_map,
            iteration=iteration,
            abs_parent_path="$",
            ref_parent_path="$",
        )

        assert mock_async_validate.call_count == 2
        mock_async_validate.assert_has_calls(
            [
                call(
                    "mock-child-1",
                    {
                        "mock-shared-metadata": "shared-metadata",
                    },
                    validator_map,
                    iteration,
                    "$.0",
                    "$.*",
                    stream=False,
                ),
                call(
                    "mock-child-2",
                    {
                        "mock-shared-metadata": "shared-metadata",
                    },
                    validator_map,
                    iteration,
                    "$.1",
                    "$.*",
                    stream=False,
                ),
            ]
        )

        assert value == ["mock-child-1-value", "mock-child-2-value"]
        assert metadata == {
            "mock-child-1-metadata": "child-1-metadata",
            "mock-child-2-metadata": "child-2-metadata",
            # NOTE: This is overriden based on who finishes last
            "mock-shared-metadata": "shared-metadata-2",
        }

    @pytest.mark.asyncio
    async def test_with_dictionary(self, mocker):
        mock_async_validate = mocker.patch.object(
            avs,
            "async_validate",
            side_effect=[
                (
                    "mock-child-1-value",
                    {
                        "mock-child-1-metadata": "child-1-metadata",
                        "mock-shared-metadata": "shared-metadata-1",
                    },
                ),
                (
                    "mock-child-2-value",
                    {
                        "mock-child-2-metadata": "child-2-metadata",
                        "mock-shared-metadata": "shared-metadata-2",
                    },
                ),
            ],
        )

        iteration = Iteration(
            call_id="mock-call",
            index=0,
        )

        validator_map = (
            {
                "$.child-1": [MagicMock(spec=Validator)],
                "$.child-2": [MagicMock(spec=Validator)],
            },
        )
        value, metadata = await avs.validate_children(
            value={"child-1": "mock-child-1", "child-2": "mock-child-2"},
            metadata={"mock-shared-metadata": "shared-metadata"},
            validator_map=validator_map,
            iteration=iteration,
            abs_parent_path="$",
            ref_parent_path="$",
        )

        assert mock_async_validate.call_count == 2
        mock_async_validate.assert_has_calls(
            [
                call(
                    "mock-child-1",
                    {
                        "mock-shared-metadata": "shared-metadata",
                    },
                    validator_map,
                    iteration,
                    "$.child-1",
                    "$.child-1",
                    stream=False,
                ),
                call(
                    "mock-child-2",
                    {
                        "mock-shared-metadata": "shared-metadata",
                    },
                    validator_map,
                    iteration,
                    "$.child-2",
                    "$.child-2",
                    stream=False,
                ),
            ]
        )

        assert value == {
            "child-1": "mock-child-1-value",
            "child-2": "mock-child-2-value",
        }
        assert metadata == {
            "mock-child-1-metadata": "child-1-metadata",
            "mock-child-2-metadata": "child-2-metadata",
            # NOTE: This is overriden based on who finishes last
            "mock-shared-metadata": "shared-metadata-2",
        }


class TestRunValidators:
    @pytest.mark.asyncio
    async def test_filter_exits_early(self, mocker):
        mock_run_validator = mocker.patch.object(
            avs,
            "run_validator",
            side_effect=[
                ValidatorRun(
                    value="mock-value",
                    metadata={},
                    on_fail_action="noop",
                    validator_logs=ValidatorLogs(
                        registered_name="noop_validator",
                        validator_name="noop_validator",
                        value_before_validation="mock-value",
                        validation_result=PassResult(),
                        property_path="$",
                    ),
                ),
                ValidatorRun(
                    value=Filter(),
                    metadata={},
                    on_fail_action="filter",
                    validator_logs=ValidatorLogs(
                        registered_name="filter_validator",
                        validator_name="filter_validator",
                        value_before_validation="mock-value",
                        validation_result=FailResult(error_message="mock-error"),
                        property_path="$",
                    ),
                ),
            ],
        )
        mock_merge_results = mocker.patch.object(avs, "merge_results")

        iteration = Iteration(
            call_id="mock-call",
            index=0,
        )

        value, metadata = await avs.run_validators(
            iteration=iteration,
            validator_map={
                "$": [
                    MagicMock(spec=Validator),
                    MagicMock(spec=Validator),
                ]
            },
            value=True,
            metadata={},
            absolute_property_path="$",
            reference_property_path="$",
        )

        assert mock_run_validator.call_count == 2
        assert mock_merge_results.call_count == 0

        assert isinstance(value, Filter)
        assert metadata == {}

    @pytest.mark.asyncio
    async def test_calls_merge(self, mocker):
        mock_run_validator = mocker.patch.object(
            avs,
            "run_validator",
            side_effect=[
                ValidatorRun(
                    value="mock-value",
                    metadata={},
                    on_fail_action="noop",
                    validator_logs=ValidatorLogs(
                        registered_name="noop_validator",
                        validator_name="noop_validator",
                        value_before_validation="mock-value",
                        validation_result=PassResult(),
                        property_path="$",
                    ),
                ),
                ValidatorRun(
                    value="mock-fix-value",
                    metadata={},
                    on_fail_action="fix",
                    validator_logs=ValidatorLogs(
                        registered_name="fix_validator",
                        validator_name="fix_validator",
                        value_before_validation="mock-value",
                        validation_result=FailResult(
                            error_message="mock-error", fix_value="mock-fix-value"
                        ),
                        property_path="$",
                    ),
                ),
            ],
        )
        mock_merge_results = mocker.patch.object(
            avs, "merge_results", return_value="mock-fix-value"
        )

        iteration = Iteration(
            call_id="mock-call",
            index=0,
        )

        value, metadata = await avs.run_validators(
            iteration=iteration,
            validator_map={
                "$": [
                    MagicMock(spec=Validator),
                    MagicMock(spec=Validator),
                ]
            },
            value=True,
            metadata={},
            absolute_property_path="$",
            reference_property_path="$",
        )

        assert mock_run_validator.call_count == 2
        assert mock_merge_results.call_count == 1

        assert value == "mock-fix-value"
        assert metadata == {}

    @pytest.mark.asyncio
    async def test_returns_value_if_no_results(self, mocker):
        mock_run_validator = mocker.patch.object(avs, "run_validator")
        mock_merge_results = mocker.patch.object(avs, "merge_results")

        iteration = Iteration(
            call_id="mock-call",
            index=0,
        )

        value, metadata = await avs.run_validators(
            iteration=iteration,
            validator_map={},
            value=True,
            metadata={},
            absolute_property_path="$",
            reference_property_path="$",
        )

        assert mock_run_validator.call_count == 0
        assert mock_merge_results.call_count == 0

        assert value is True
        assert metadata == {}


class TestRunValidator:
    @pytest.mark.asyncio
    async def test_pass_result(self, mocker):
        validator_logs = ValidatorLogs(
            validator_name="mock-validator",
            registered_name="mock-validator",
            instance_id=1234,
            property_path="$",
            value_before_validation="value",
            start_time=datetime(2024, 9, 10, 9, 54, 0, 38391),
            value_after_validation="value",
        )
        mock_before_run_validator = mocker.patch.object(
            avs, "before_run_validator", return_value=validator_logs
        )

        validation_result = PassResult()
        mock_run_validator_async = mocker.patch.object(
            avs, "run_validator_async", return_value=validation_result
        )

        mock_after_run_validator = mocker.patch.object(
            avs, "after_run_validator", return_value=validator_logs
        )

        iteration = Iteration(
            call_id="mock-call",
            index=0,
        )
        validator = MagicMock(spec=Validator)
        validator.on_fail_descriptor = "noop"

        result = await avs.run_validator(
            iteration=iteration,
            validator=validator,
            value="value",
            metadata={},
            absolute_property_path="$",
        )

        assert mock_before_run_validator.call_count == 1
        mock_before_run_validator.assert_called_once_with(iteration, validator, "value", "$")

        assert mock_run_validator_async.call_count == 1
        mock_run_validator_async.assert_called_once_with(
            validator,
            "value",
            {},
            False,
            validation_session_id=iteration.id,
            reference_path=None,
        )

        assert mock_after_run_validator.call_count == 1
        mock_after_run_validator.assert_called_once_with(
            validator, validator_logs, validation_result
        )

        assert isinstance(result, ValidatorRun)
        assert result.value == "value"
        assert result.metadata == {}
        assert result.validator_logs == validator_logs

    @pytest.mark.asyncio
    async def test_pass_result_with_override(self, mocker):
        validator_logs = ValidatorLogs(
            validator_name="mock-validator",
            registered_name="mock-validator",
            instance_id=1234,
            property_path="$",
            value_before_validation="value",
            start_time=datetime(2024, 9, 10, 9, 54, 0, 38391),
            value_after_validation="value",
        )
        mock_before_run_validator = mocker.patch.object(
            avs, "before_run_validator", return_value=validator_logs
        )

        validation_result = PassResult(value_override="override")
        mock_run_validator_async = mocker.patch.object(
            avs, "run_validator_async", return_value=validation_result
        )

        mock_after_run_validator = mocker.patch.object(
            avs, "after_run_validator", return_value=validator_logs
        )

        iteration = Iteration(
            call_id="mock-call",
            index=0,
        )
        validator = MagicMock(spec=Validator)
        validator.on_fail_descriptor = "noop"

        result = await avs.run_validator(
            iteration=iteration,
            validator=validator,
            value="value",
            metadata={},
            absolute_property_path="$",
        )

        assert mock_before_run_validator.call_count == 1
        mock_before_run_validator.assert_called_once_with(iteration, validator, "value", "$")

        assert mock_run_validator_async.call_count == 1
        mock_run_validator_async.assert_called_once_with(
            validator,
            "value",
            {},
            False,
            validation_session_id=iteration.id,
            reference_path=None,
        )

        assert mock_after_run_validator.call_count == 1
        mock_after_run_validator.assert_called_once_with(
            validator, validator_logs, validation_result
        )

        assert isinstance(result, ValidatorRun)
        assert result.value == "override"
        assert result.metadata == {}
        assert result.validator_logs == validator_logs

    @pytest.mark.asyncio
    async def test_fail_result(self, mocker):
        validator_logs = ValidatorLogs(
            validator_name="mock-validator",
            registered_name="mock-validator",
            instance_id=1234,
            property_path="$",
            value_before_validation="value",
            start_time=datetime(2024, 9, 10, 9, 54, 0, 38391),
            value_after_validation="value",
        )
        mock_before_run_validator = mocker.patch.object(
            avs, "before_run_validator", return_value=validator_logs
        )

        validation_result = FailResult(error_message="mock-error")
        mock_run_validator_async = mocker.patch.object(
            avs, "run_validator_async", return_value=validation_result
        )

        mock_after_run_validator = mocker.patch.object(
            avs, "after_run_validator", return_value=validator_logs
        )

        mock_perform_correction = mocker.patch.object(
            avs, "perform_correction", return_value="corrected-value"
        )

        iteration = Iteration(
            call_id="mock-call",
            index=0,
        )
        validator = MagicMock(spec=Validator)
        validator.on_fail_descriptor = "noop"

        result = await avs.run_validator(
            iteration=iteration,
            validator=validator,
            value="value",
            metadata={},
            absolute_property_path="$",
        )

        assert mock_before_run_validator.call_count == 1
        mock_before_run_validator.assert_called_once_with(iteration, validator, "value", "$")

        assert mock_run_validator_async.call_count == 1
        mock_run_validator_async.assert_called_once_with(
            validator,
            "value",
            {},
            False,
            validation_session_id=iteration.id,
            reference_path=None,
        )

        assert mock_after_run_validator.call_count == 1
        mock_after_run_validator.assert_called_once_with(
            validator, validator_logs, validation_result
        )

        assert mock_perform_correction.call_count == 1
        mock_perform_correction.assert_called_once_with(
            validation_result, "value", validator, rechecked_value=None
        )

        assert isinstance(result, ValidatorRun)
        assert result.value == "corrected-value"
        assert result.metadata == {}
        assert result.validator_logs == validator_logs

    @pytest.mark.asyncio
    async def test_fail_result_with_fix_reask(self, mocker):
        validator_logs = ValidatorLogs(
            validator_name="mock-validator",
            registered_name="mock-validator",
            instance_id=1234,
            property_path="$",
            value_before_validation="value",
            start_time=datetime(2024, 9, 10, 9, 54, 0, 38391),
            value_after_validation="value",
        )
        mock_before_run_validator = mocker.patch.object(
            avs, "before_run_validator", return_value=validator_logs
        )

        validation_result = FailResult(error_message="mock-error", fix_value="fixed-value")
        rechecked_result = PassResult()
        mock_run_validator_async = mocker.patch.object(
            avs,
            "run_validator_async",
            side_effect=[validation_result, rechecked_result],
        )

        mock_after_run_validator = mocker.patch.object(
            avs, "after_run_validator", return_value=validator_logs
        )

        mock_perform_correction = mocker.patch.object(
            avs, "perform_correction", return_value="fixed-value"
        )

        iteration = Iteration(
            call_id="mock-call",
            index=0,
        )
        validator = MagicMock(spec=Validator)
        validator.on_fail_descriptor = OnFailAction.FIX_REASK

        result = await avs.run_validator(
            iteration=iteration,
            validator=validator,
            value="value",
            metadata={},
            absolute_property_path="$",
        )

        assert mock_before_run_validator.call_count == 1
        mock_before_run_validator.assert_called_once_with(iteration, validator, "value", "$")

        assert mock_run_validator_async.call_count == 2
        mock_run_validator_async.assert_has_calls(
            [
                call(
                    validator,
                    "value",
                    {},
                    False,
                    validation_session_id=iteration.id,
                    reference_path=None,
                ),
                call(
                    validator,
                    "fixed-value",
                    {},
                    False,
                    validation_session_id=iteration.id,
                    reference_path=None,
                ),
            ]
        )

        assert mock_after_run_validator.call_count == 1
        mock_after_run_validator.assert_called_once_with(
            validator, validator_logs, validation_result
        )

        assert mock_perform_correction.call_count == 1
        mock_perform_correction.assert_called_once_with(
            validation_result, "value", validator, rechecked_value=rechecked_result
        )

        assert isinstance(result, ValidatorRun)
        assert result.value == "fixed-value"
        assert result.metadata == {}
        assert result.validator_logs == validator_logs


class TestRunValidatorAsync:
    @pytest.mark.asyncio
    async def test_happy_path(self, mocker):
        mock_validator = MagicMock(spec=Validator)

        validation_result = PassResult()
        mock_execute_validator = mocker.patch.object(
            avs, "execute_validator", return_value=validation_result
        )

        result = await avs.run_validator_async(
            validator=mock_validator,
            value="value",
            metadata={},
            stream=False,
            validation_session_id="mock-session",
        )

        assert result == validation_result

        assert mock_execute_validator.call_count == 1
        mock_execute_validator.assert_called_once_with(
            mock_validator, "value", {}, False, validation_session_id="mock-session"
        )

    @pytest.mark.asyncio
    async def test_result_is_none(self, mocker):
        mock_validator = MagicMock(spec=Validator)

        validation_result = None
        mock_execute_validator = mocker.patch.object(
            avs, "execute_validator", return_value=validation_result
        )

        result = await avs.run_validator_async(
            validator=mock_validator,
            value="value",
            metadata={},
            stream=False,
            validation_session_id="mock-session",
        )

        assert isinstance(result, PassResult)

        assert mock_execute_validator.call_count == 1
        mock_execute_validator.assert_called_once_with(
            mock_validator, "value", {}, False, validation_session_id="mock-session"
        )
