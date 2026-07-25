import asyncio
from asyncio.unix_events import _UnixSelectorEventLoop
import os
import pytest

from guardrails.validator_service import should_run_sync, get_loop
from guardrails.classes.history import Iteration
from guardrails.validator_base import Validator, register_validator


try:
    import uvloop
except ImportError:
    uvloop = None


def get_event_loop():
    # Python 3.13+ raises if there is no current event loop in the main
    # thread instead of creating one, so create and register one explicitly.
    try:
        return asyncio.get_event_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        return loop


class TestShouldRunSync:
    def test_guardrails_run_sync_is_true(self):
        GUARDRAILS_RUN_SYNC_bak = os.environ.get("GUARDRAILS_RUN_SYNC")
        os.environ["GUARDRAILS_RUN_SYNC"] = "true"

        result = should_run_sync()
        assert result is True

        if GUARDRAILS_RUN_SYNC_bak is not None:
            os.environ["GUARDRAILS_RUN_SYNC"] = GUARDRAILS_RUN_SYNC_bak
        else:
            del os.environ["GUARDRAILS_RUN_SYNC"]

    def test_guardrails_run_sync_is_false(self):
        GUARDRAILS_RUN_SYNC_bak = os.environ.get("GUARDRAILS_RUN_SYNC")
        os.environ["GUARDRAILS_RUN_SYNC"] = "false"

        result = should_run_sync()
        assert result is False

        if GUARDRAILS_RUN_SYNC_bak is not None:
            os.environ["GUARDRAILS_RUN_SYNC"] = GUARDRAILS_RUN_SYNC_bak
        else:
            del os.environ["GUARDRAILS_RUN_SYNC"]


class TestGetLoop:
    def test_raises_if_loop_is_running(self):
        loop = get_event_loop()

        async def callback():
            # NOTE: This means only AsyncGuard will parallelize validators
            #       if it's called within an async function.
            with pytest.raises(RuntimeError, match="An event loop is already running."):
                get_loop()

        loop.run_until_complete(callback())

    @pytest.mark.skipif(uvloop is None, reason="uvloop is not installed")
    def test_uvloop_is_used_when_installed(self):
        loop = get_loop()
        assert isinstance(loop, uvloop.Loop)

    @pytest.mark.skipif(uvloop is not None, reason="uvloop is installed")
    def test_asyncio_default_is_used_otherwise(self):
        loop = get_loop()
        assert isinstance(loop, _UnixSelectorEventLoop)


class TestValidate:
    def test_forced_sync(self, mocker):
        GUARDRAILS_RUN_SYNC_bak = os.environ.get("GUARDRAILS_RUN_SYNC")
        os.environ["GUARDRAILS_RUN_SYNC"] = "true"

        from guardrails.validator_service import validate, SequentialValidatorService

        mocker.spy(SequentialValidatorService, "__init__")
        mocker.spy(SequentialValidatorService, "validate")

        iteration = Iteration(
            call_id="mock_call_id",
            index=0,
        )

        value, metadata = validate(
            value="value",
            metadata={},
            validator_map={},
            iteration=iteration,
        )

        assert value == "value"
        assert metadata == {}
        SequentialValidatorService.__init__.assert_called_once()
        SequentialValidatorService.validate.assert_called_once()

        if GUARDRAILS_RUN_SYNC_bak is not None:
            os.environ["GUARDRAILS_RUN_SYNC"] = GUARDRAILS_RUN_SYNC_bak
        else:
            del os.environ["GUARDRAILS_RUN_SYNC"]

    def test_async(self, mocker):
        from guardrails.validator_service import validate, AsyncValidatorService

        mocker.spy(AsyncValidatorService, "__init__")
        mocker.spy(AsyncValidatorService, "validate")

        iteration = Iteration(
            call_id="mock_call_id",
            index=0,
        )

        value, metadata = validate(
            value="value",
            metadata={},
            validator_map={},
            iteration=iteration,
        )

        assert value == "value"
        assert metadata == {}
        AsyncValidatorService.__init__.assert_called_once()
        AsyncValidatorService.validate.assert_called_once()

    def test_sync_busy_loop(self, mocker):
        from guardrails.validator_service import validate, SequentialValidatorService

        mocker.spy(SequentialValidatorService, "__init__")
        mocker.spy(SequentialValidatorService, "validate")

        iteration = Iteration(
            call_id="mock_call_id",
            index=0,
        )

        loop = get_event_loop()

        async def callback():
            with pytest.warns(
                Warning,
                match=(
                    "Could not obtain an event loop."
                    " Falling back to synchronous validation."
                ),
            ):
                value, metadata = validate(
                    value="value",
                    metadata={},
                    validator_map={},
                    iteration=iteration,
                )
                assert value == "value"
                assert metadata == {}

        loop.run_until_complete(callback())

        SequentialValidatorService.__init__.assert_called_once()
        SequentialValidatorService.validate.assert_called_once()


@register_validator(name="tests/returns-none", data_type="string")
class ReturnsNone(Validator):
    """A malformed validator: _validate() returns neither Pass nor Fail."""

    def _validate(self, value, metadata):
        return None


@pytest.mark.parametrize("run_sync", ["true", "false"])
def test_non_stream_none_result_errors_in_both_services(monkeypatch, run_sync):
    # Both validator services must reject an undocumented non-stream result
    # type rather than one of them promoting it to a pass.
    monkeypatch.setenv("GUARDRAILS_RUN_SYNC", run_sync)

    from guardrails.validator_service import validate

    iteration = Iteration(
        call_id="mock_call_id",
        index=0,
    )

    with pytest.raises(RuntimeError, match="Unexpected result type"):
        validate(
            value="value",
            metadata={},
            validator_map={"$": [ReturnsNone()]},
            iteration=iteration,
        )


@pytest.mark.asyncio
async def test_async_validate(mocker):
    from guardrails.validator_service import async_validate, AsyncValidatorService

    mocker.spy(AsyncValidatorService, "__init__")
    mocker.spy(AsyncValidatorService, "async_validate")

    iteration = Iteration(
        call_id="mock_call_id",
        index=0,
    )

    value, metadata = await async_validate(
        value="value",
        metadata={},
        validator_map={},
        iteration=iteration,
    )

    assert value == "value"
    assert metadata == {}
    AsyncValidatorService.__init__.assert_called_once()
    AsyncValidatorService.async_validate.assert_called_once()
