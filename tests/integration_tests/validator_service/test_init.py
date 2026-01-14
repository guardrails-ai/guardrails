from asyncio import get_event_loop
from asyncio.unix_events import _UnixSelectorEventLoop
import os
import pytest

from guardrails.validator_service import should_run_sync, get_loop
from guardrails.classes.history import Iteration


try:
    import uvloop
except ImportError:
    uvloop = None


class TestShouldRunSync:
    def test_process_count_is_one(self):
        GUARDRAILS_PROCESS_COUNT_bak = os.environ.get("GUARDRAILS_PROCESS_COUNT")
        GUARDRAILS_RUN_SYNC_bak = os.environ.get("GUARDRAILS_RUN_SYNC")
        os.environ["GUARDRAILS_PROCESS_COUNT"] = "1"
        if os.environ.get("GUARDRAILS_RUN_SYNC"):
            del os.environ["GUARDRAILS_RUN_SYNC"]

        with pytest.warns(
            DeprecationWarning,
            match=(
                "GUARDRAILS_PROCESS_COUNT is deprecated"
                " and will be removed in a future release."
                " To force synchronous validation,"
                " please use GUARDRAILS_RUN_SYNC instead."
            ),
        ):
            result = should_run_sync()
            assert result is True

        if GUARDRAILS_PROCESS_COUNT_bak is not None:
            os.environ["GUARDRAILS_PROCESS_COUNT"] = GUARDRAILS_PROCESS_COUNT_bak
        else:
            del os.environ["GUARDRAILS_PROCESS_COUNT"]
        if GUARDRAILS_RUN_SYNC_bak is not None:
            os.environ["GUARDRAILS_RUN_SYNC"] = GUARDRAILS_RUN_SYNC_bak

    def test_process_count_is_2(self):
        GUARDRAILS_PROCESS_COUNT_bak = os.environ.get("GUARDRAILS_PROCESS_COUNT")
        GUARDRAILS_RUN_SYNC_bak = os.environ.get("GUARDRAILS_RUN_SYNC")
        os.environ["GUARDRAILS_PROCESS_COUNT"] = "2"
        if os.environ.get("GUARDRAILS_RUN_SYNC"):
            del os.environ["GUARDRAILS_RUN_SYNC"]

        with pytest.warns(
            DeprecationWarning,
            match=(
                "GUARDRAILS_PROCESS_COUNT is deprecated"
                " and will be removed in a future release."
                " To force synchronous validation,"
                " please use GUARDRAILS_RUN_SYNC instead."
            ),
        ):
            result = should_run_sync()
            assert result is False

        if GUARDRAILS_PROCESS_COUNT_bak is not None:
            os.environ["GUARDRAILS_PROCESS_COUNT"] = GUARDRAILS_PROCESS_COUNT_bak
        else:
            del os.environ["GUARDRAILS_PROCESS_COUNT"]
        if GUARDRAILS_RUN_SYNC_bak is not None:
            os.environ["GUARDRAILS_RUN_SYNC"] = GUARDRAILS_RUN_SYNC_bak

    def test_guardrails_run_sync_is_true(self):
        GUARDRAILS_PROCESS_COUNT_bak = os.environ.get("GUARDRAILS_PROCESS_COUNT")
        GUARDRAILS_RUN_SYNC_bak = os.environ.get("GUARDRAILS_RUN_SYNC")
        os.environ["GUARDRAILS_RUN_SYNC"] = "true"
        if os.environ.get("GUARDRAILS_PROCESS_COUNT"):
            del os.environ["GUARDRAILS_PROCESS_COUNT"]

        result = should_run_sync()
        assert result is True

        if GUARDRAILS_PROCESS_COUNT_bak is not None:
            os.environ["GUARDRAILS_PROCESS_COUNT"] = GUARDRAILS_PROCESS_COUNT_bak
        if GUARDRAILS_RUN_SYNC_bak is not None:
            os.environ["GUARDRAILS_RUN_SYNC"] = GUARDRAILS_RUN_SYNC_bak
        else:
            del os.environ["GUARDRAILS_RUN_SYNC"]

    def test_guardrails_run_sync_is_false(self):
        GUARDRAILS_PROCESS_COUNT_bak = os.environ.get("GUARDRAILS_PROCESS_COUNT")
        GUARDRAILS_RUN_SYNC_bak = os.environ.get("GUARDRAILS_RUN_SYNC")
        os.environ["GUARDRAILS_RUN_SYNC"] = "false"
        if os.environ.get("GUARDRAILS_PROCESS_COUNT"):
            del os.environ["GUARDRAILS_PROCESS_COUNT"]

        result = should_run_sync()
        assert result is False

        if GUARDRAILS_PROCESS_COUNT_bak is not None:
            os.environ["GUARDRAILS_PROCESS_COUNT"] = GUARDRAILS_PROCESS_COUNT_bak
        if GUARDRAILS_RUN_SYNC_bak is not None:
            os.environ["GUARDRAILS_RUN_SYNC"] = GUARDRAILS_RUN_SYNC_bak
        else:
            del os.environ["GUARDRAILS_RUN_SYNC"]

    def test_process_count_is_1_and_guardrails_run_sync_is_false(self):
        GUARDRAILS_PROCESS_COUNT_bak = os.environ.get("GUARDRAILS_PROCESS_COUNT")
        GUARDRAILS_RUN_SYNC_bak = os.environ.get("GUARDRAILS_RUN_SYNC")
        os.environ["GUARDRAILS_PROCESS_COUNT"] = "1"
        os.environ["GUARDRAILS_RUN_SYNC"] = "false"

        with pytest.warns(
            DeprecationWarning,
            match=(
                "GUARDRAILS_PROCESS_COUNT is deprecated"
                " and will be removed in a future release."
                " To force synchronous validation,"
                " please use GUARDRAILS_RUN_SYNC instead."
            ),
        ):
            result = should_run_sync()
            assert result is True

        if GUARDRAILS_PROCESS_COUNT_bak is not None:
            os.environ["GUARDRAILS_PROCESS_COUNT"] = GUARDRAILS_PROCESS_COUNT_bak
        else:
            del os.environ["GUARDRAILS_PROCESS_COUNT"]
        if GUARDRAILS_RUN_SYNC_bak is not None:
            os.environ["GUARDRAILS_RUN_SYNC"] = GUARDRAILS_RUN_SYNC_bak
        else:
            del os.environ["GUARDRAILS_RUN_SYNC"]

    def test_process_count_is_2_and_guardrails_run_sync_is_true(self):
        GUARDRAILS_PROCESS_COUNT_bak = os.environ.get("GUARDRAILS_PROCESS_COUNT")
        GUARDRAILS_RUN_SYNC_bak = os.environ.get("GUARDRAILS_RUN_SYNC")
        os.environ["GUARDRAILS_PROCESS_COUNT"] = "2"
        os.environ["GUARDRAILS_RUN_SYNC"] = "true"

        with pytest.warns(
            DeprecationWarning,
            match=(
                "GUARDRAILS_PROCESS_COUNT is deprecated"
                " and will be removed in a future release."
                " To force synchronous validation,"
                " please use GUARDRAILS_RUN_SYNC instead."
            ),
        ):
            result = should_run_sync()
            assert result is True

        if GUARDRAILS_PROCESS_COUNT_bak is not None:
            os.environ["GUARDRAILS_PROCESS_COUNT"] = GUARDRAILS_PROCESS_COUNT_bak
        else:
            del os.environ["GUARDRAILS_PROCESS_COUNT"]
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
        GUARDRAILS_PROCESS_COUNT_bak = os.environ.get("GUARDRAILS_PROCESS_COUNT")
        GUARDRAILS_RUN_SYNC_bak = os.environ.get("GUARDRAILS_RUN_SYNC")
        os.environ["GUARDRAILS_RUN_SYNC"] = "true"
        if os.environ.get("GUARDRAILS_PROCESS_COUNT"):
            del os.environ["GUARDRAILS_PROCESS_COUNT"]

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

        if GUARDRAILS_PROCESS_COUNT_bak is not None:
            os.environ["GUARDRAILS_PROCESS_COUNT"] = GUARDRAILS_PROCESS_COUNT_bak
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
                match=("Could not obtain an event loop. Falling back to synchronous validation."),
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
