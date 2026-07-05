"""Tests for SQLiteTraceHandler._truncate throttling."""

from unittest.mock import MagicMock

from guardrails.call_tracing.sqlite_trace_handler import SQLiteTraceHandler


def test_truncate_throttled_within_cleanup_interval(tmp_path):
    """_truncate should only execute DELETE when the cleanup interval has
    elapsed (or force=True), not on every call.

    Regression test: the DELETE statement was outside the ``if`` block,
    causing it to run on every call and defeating the TIME_BETWEEN_CLEANUPS
    throttle.
    """
    handler = SQLiteTraceHandler(tmp_path / "test.db", read_mode=False)
    # Replace db with a mock so we can count execute calls without a real
    # sqlite3.Connection (whose .execute attribute is read-only).
    handler.db = MagicMock()

    # __init__ sets last_cleanup = time.time(), so immediately after
    # construction the interval has NOT elapsed → DELETE should NOT run.
    handler._truncate()
    assert handler.db.execute.call_count == 0, (
        "DELETE should not run within the cleanup interval"
    )

    # force=True bypasses the throttle → DELETE should run.
    handler._truncate(force=True)
    assert handler.db.execute.call_count == 1, "DELETE should run when force=True"

    # Another force call → DELETE runs again.
    handler._truncate(force=True)
    assert handler.db.execute.call_count == 2

    # Non-force immediately after → throttled, no new DELETE.
    handler._truncate()
    assert handler.db.execute.call_count == 2, (
        "DELETE should be throttled within the interval"
    )
