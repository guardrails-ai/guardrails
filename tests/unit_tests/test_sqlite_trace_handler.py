from unittest.mock import MagicMock

import pytest

from guardrails.call_tracing.sqlite_trace_handler import SQLiteTraceHandler


@pytest.fixture
def handler(tmp_path):
    handler = SQLiteTraceHandler(tmp_path / "test.db", read_mode=False)
    handler.db = MagicMock()
    return handler


def test_truncate_is_throttled_within_cleanup_interval(handler):
    handler._truncate()

    assert handler.db.execute.call_count == 0, (
        "a DELETE must not run before TIME_BETWEEN_CLEANUPS has elapsed"
    )


def test_truncate_runs_after_interval_elapses(handler):
    handler.last_cleanup = 0

    handler._truncate()

    assert handler.db.execute.call_count == 1


def test_truncate_force_runs_delete(handler):
    handler._truncate(force=True)

    assert handler.db.execute.call_count == 1
