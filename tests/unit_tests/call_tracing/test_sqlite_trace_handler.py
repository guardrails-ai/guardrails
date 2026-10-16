from guardrails.call_tracing.sqlite_trace_handler import SQLiteTraceHandler


def test_tail_logs_follow_does_not_repeat_out_of_order_rows(tmp_path):
    log_path = tmp_path / "guardrails_calls.db"
    writer = SQLiteTraceHandler(log_path, read_mode=False)

    writer.log("first", 2.0, 3.0, "first", "first", "")
    writer.log("second", 1.0, 2.0, "second", "second", "")

    reader = SQLiteTraceHandler(log_path, read_mode=True)
    logs = reader.tail_logs(follow=True)

    try:
        first_batch = [next(logs), next(logs)]
        writer.log("third", 3.0, 4.0, "third", "third", "")
        next_entry = next(logs)
    finally:
        reader.db.close()
        writer.db.close()

    assert [entry.guard_name for entry in first_batch] == ["second", "first"]
    assert next_entry.guard_name == "third"
