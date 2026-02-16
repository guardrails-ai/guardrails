from unittest.mock import AsyncMock
import pytest

import guardrails.validator_service as vs
from guardrails.classes.history.iteration import Iteration


iteration = Iteration(
    call_id="mock-call",
    index=0,
)


class TestShouldRunSync:
    def test_run_sync_set_to_true(self, mocker):
        mocker.patch(
            "guardrails.validator_service.os.environ.get", side_effect=["True"]
        )
        assert vs.should_run_sync() is True

    def test_should_run_sync_default(self, mocker):
        mocker.patch(
            "guardrails.validator_service.os.environ.get", side_effect=["false"]
        )
        assert vs.should_run_sync() is False


class TestGetLoop:
    def test_get_loop_with_running_loop(self, mocker):
        mocker.patch(
            "guardrails.validator_service.asyncio.get_running_loop",
            return_value="running loop",
        )
        with pytest.raises(RuntimeError):
            vs.get_loop()

    def test_get_loop_without_running_loop(self, mocker):
        mocker.patch(
            "guardrails.validator_service.asyncio.get_running_loop",
            side_effect=RuntimeError,
        )
        mocker.patch(
            "guardrails.validator_service.asyncio.get_event_loop",
            return_value="event loop",
        )
        assert vs.get_loop() == "event loop"

    def test_get_loop_with_uvloop(self, mocker):
        mocker.patch("guardrails.validator_service.uvloop")
        mock_event_loop_policy = mocker.patch(
            "guardrails.validator_service.uvloop.EventLoopPolicy"
        )
        mocker.patch(
            "guardrails.validator_service.asyncio.get_running_loop",
            side_effect=RuntimeError,
        )
        mocker.patch(
            "guardrails.validator_service.asyncio.get_event_loop",
            return_value="event loop",
        )
        mock_set_event_loop_policy = mocker.patch("asyncio.set_event_loop_policy")

        assert vs.get_loop() == "event loop"

        mock_event_loop_policy.assert_called_once()
        mock_set_event_loop_policy.assert_called_once_with(
            mock_event_loop_policy.return_value
        )


class TestValidate:
    def test_validate_with_sync(self, mocker):
        mocker.patch("guardrails.validator_service.should_run_sync", return_value=True)
        mocker.patch("guardrails.validator_service.SequentialValidatorService")
        mocker.patch("guardrails.validator_service.AsyncValidatorService")
        mocker.patch("guardrails.validator_service.get_loop")
        mocker.patch("guardrails.validator_service.warnings")

        vs.validate(
            value=True,
            metadata={},
            validator_map={},
            iteration=iteration,
        )

        vs.SequentialValidatorService.assert_called_once_with(True)
        vs.SequentialValidatorService.return_value.validate.assert_called_once_with(
            True,
            {},
            {},
            iteration,
            "$",
            "$",
            loop=None,
        )

    def test_validate_with_async(self, mocker):
        mocker.patch("guardrails.validator_service.should_run_sync", return_value=False)
        mocker.patch("guardrails.validator_service.SequentialValidatorService")
        mocker.patch("guardrails.validator_service.AsyncValidatorService")
        mocker.patch("guardrails.validator_service.get_loop", return_value="event loop")
        mocker.patch("guardrails.validator_service.warnings")

        vs.validate(
            value=True,
            metadata={},
            validator_map={},
            iteration=iteration,
        )

        vs.AsyncValidatorService.assert_called_once_with(True)
        vs.AsyncValidatorService.return_value.validate.assert_called_once_with(
            True,
            {},
            {},
            iteration,
            "$",
            "$",
            loop="event loop",
        )

    def test_validate_with_no_available_event_loop(self, mocker):
        mocker.patch("guardrails.validator_service.should_run_sync", return_value=False)
        mocker.patch("guardrails.validator_service.SequentialValidatorService")
        mocker.patch("guardrails.validator_service.AsyncValidatorService")
        mocker.patch("guardrails.validator_service.get_loop", side_effect=RuntimeError)
        mock_warn = mocker.patch("guardrails.validator_service.warnings.warn")

        vs.validate(
            value=True,
            metadata={},
            validator_map={},
            iteration=iteration,
        )

        mock_warn.assert_called_once_with(
            "Could not obtain an event loop. Falling back to synchronous validation."
        )

        vs.SequentialValidatorService.assert_called_once_with(True)
        vs.SequentialValidatorService.return_value.validate.assert_called_once_with(
            True,
            {},
            {},
            iteration,
            "$",
            "$",
            loop=None,
        )


@pytest.mark.asyncio
async def test_async_validate(mocker):
    mocker.patch(
        "guardrails.validator_service.AsyncValidatorService", return_value=AsyncMock()
    )
    await vs.async_validate(
        value=True,
        metadata={},
        validator_map={},
        iteration=iteration,
    )

    vs.AsyncValidatorService.assert_called_once_with(True)
    vs.AsyncValidatorService.return_value.async_validate.assert_called_once_with(
        True, {}, {}, iteration, "$", "$", False
    )
