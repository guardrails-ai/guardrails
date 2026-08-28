import json

import pytest

from guardrails.telemetry.open_inference import trace_llm_call


@pytest.fixture
def captured_span_attributes(monkeypatch):
    class CapturingSpan:
        def __init__(self):
            self.attributes = {}

        def set_attribute(self, key, value):
            self.attributes[key] = value

    def record(**kwargs):
        span = CapturingSpan()
        monkeypatch.setattr(
            "guardrails.telemetry.open_inference.get_span", lambda: span
        )
        trace_llm_call(**kwargs)
        return span.attributes

    return record


def test_invocation_parameters_are_valid_json(captured_span_attributes):
    attrs = captured_span_attributes(
        invocation_parameters={
            "temperature": 0.3,
            "model": "gpt-4o-mini",
            "messages": [{"role": "user", "content": "hi"}],
        }
    )

    assert json.loads(attrs["llm.invocation_parameters"]) == {
        "temperature": 0.3,
        "model": "gpt-4o-mini",
    }


def test_prompt_payloads_are_excluded(captured_span_attributes):
    attrs = captured_span_attributes(
        invocation_parameters={
            "temperature": 0.3,
            "messages": [{"role": "user", "content": "secret prompt"}],
            "prompt": "secret prompt",
            "instructions": "system text",
            "input": "user text",
        }
    )

    assert json.loads(attrs["llm.invocation_parameters"]) == {"temperature": 0.3}


def test_per_key_parameters_are_emitted_and_redacted(captured_span_attributes):
    attrs = captured_span_attributes(
        invocation_parameters={
            "api_key": "sk-1234",
            "temperature": 0.3,
            "stream": False,
            "tools": [{"type": "function"}],
            "nullable": None,
        }
    )

    assert attrs["llm.invocation_parameters.api_key"] == "***1234"
    assert attrs["llm.invocation_parameters.temperature"] == 0.3
    assert attrs["llm.invocation_parameters.stream"] is False
    assert json.loads(attrs["llm.invocation_parameters.tools"]) == [
        {"type": "function"}
    ]
    assert "llm.invocation_parameters.nullable" not in attrs


def test_json_string_input_is_normalized(captured_span_attributes):
    attrs = captured_span_attributes(
        invocation_parameters=json.dumps({"temperature": 0.3})  # type: ignore[arg-type]
    )

    assert json.loads(attrs["llm.invocation_parameters"]) == {"temperature": 0.3}


def test_unparseable_input_sets_no_invocation_attributes(captured_span_attributes):
    attrs = captured_span_attributes(invocation_parameters=object())  # type: ignore[arg-type]

    assert not any(key.startswith("llm.invocation_parameters") for key in attrs)


def test_other_llm_attributes_are_unchanged(captured_span_attributes):
    attrs = captured_span_attributes(
        input_messages=[{"role": "user", "content": "hi"}],
        invocation_parameters={"temperature": 0.3},
        model_name="gpt-4o-mini",
    )

    assert attrs["llm.model_name"] == "gpt-4o-mini"
    assert attrs["llm.input_messages.0.message.role"] == "user"
