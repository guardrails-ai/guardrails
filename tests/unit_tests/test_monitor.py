"""Unit tests for prompt and instructions parsing."""

import os

from guardrails.monitor import Sink


def test_sink():
    # Create a temporary file with ".jsonl" suffix
    tmpfile = './tmpfile.jsonl'
    sink = Sink(tmpfile)

    # Write a line to the file
    sink.write({"test": "test"})

    # Flush the file
    sink.flush()

    # Write multiple lines to the file
    # The dicts should appear merged as a single line.
    sink.write({"test": "test"})
    sink.write({"test2": "test2"})
    sink.flush()

    # Read the file and check the contents.
    with open(tmpfile, 'r') as f:
        contents = f.read()
        assert contents == '{"test": "test"}\n{"test": "test", "test2": "test2"}\n'

    # Delete the file
    os.remove(tmpfile)