import json

import pytest

from guardrails.telemetry.open_inference import trace_llm_call


@pytest.fixture
def captured_span_attributes():
    """Register an in-memory exporter and return a callable that runs
    trace_llm_call inside a real span, yielding the recorded attributes."""
    from opentelemetry import trace as otel_trace
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import SimpleSpanProcessor
    from opentelemetry.sdk.trace.export.in_memory_span_exporter import (
        InMemorySpanExporter,
    )

    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    otel_trace.set_tracer_provider(provider)
    tracer = provider.get_tracer("test")

    def record(**kwargs):
        with tracer.start_as_current_span("call"):
            trace_llm_call(**kwargs)
        return dict(exporter.get_finished_spans()[0].attributes)

    return record


class TestInvocationParametersBlob:
    """The llm.invocation_parameters attribute must be valid JSON.

    Regression for issue #1631: recursive_key_operation re-serialized the
    JSON string input with str(), so consumers reading the attribute got a
    Python repr (single quotes) and json.loads raised JSONDecodeError.
    """

    def test_blob_is_parseable_json(self, captured_span_attributes):
        attrs = captured_span_attributes(
            invocation_parameters={
                "temperature": 0.3,
                "model": "gpt-4o-mini",
                "messages": [{"role": "user", "content": "hi"}],
            },
        )
        blob = attrs["llm.invocation_parameters"]
        parsed = json.loads(blob)  # must not raise
        assert parsed["temperature"] == 0.3
        assert parsed["model"] == "gpt-4o-mini"
        assert "'" not in blob

    def test_message_payloads_excluded(self, captured_span_attributes):
        # Messages/prompt are already carried by input.value and
        # llm.input_messages.*; duplicating them into model parameters leaks
        # prompt content into masking-unaware fields.
        attrs = captured_span_attributes(
            invocation_parameters={
                "temperature": 0.3,
                "messages": [{"role": "user", "content": "secret prompt"}],
                "prompt": "secret prompt",
                "instructions": "system text",
                "input": "user text",
            },
        )
        blob = attrs["llm.invocation_parameters"]
        assert "secret" not in blob
        assert set(json.loads(blob)) == {"temperature"}

    def test_sensitive_values_redacted_in_blob(self, captured_span_attributes):
        attrs = captured_span_attributes(
            invocation_parameters={"api_key": "sk-1234", "model": "gpt-4o-mini"},
        )
        blob = json.loads(attrs["llm.invocation_parameters"])
        assert blob["api_key"] == "***1234"

    def test_json_string_input_normalized(self, captured_span_attributes):
        attrs = captured_span_attributes(
            invocation_parameters=json.dumps({"temperature": 0.3}),
        )
        assert json.loads(attrs["llm.invocation_parameters"]) == {"temperature": 0.3}

    def test_unparseable_input_sets_no_attribute(self, captured_span_attributes):
        attrs = captured_span_attributes(
            invocation_parameters=object(),
        )
        assert not any(key.startswith("llm.invocation_parameters") for key in attrs)


class TestPerKeyInvocationParameters:
    """OpenInference consumers read llm.invocation_parameters.<name> per
    key; none were emitted before (issue #1631)."""

    def test_scalar_parameters_emitted_verbatim(self, captured_span_attributes):
        attrs = captured_span_attributes(
            invocation_parameters={
                "temperature": 0.3,
                "max_tokens": 100,
                "stream": False,
                "model": "gpt-4o-mini",
            },
        )
        assert attrs["llm.invocation_parameters.temperature"] == 0.3
        assert attrs["llm.invocation_parameters.max_tokens"] == 100
        assert attrs["llm.invocation_parameters.stream"] is False
        assert attrs["llm.invocation_parameters.model"] == "gpt-4o-mini"

    def test_structured_parameters_emitted_as_json(self, captured_span_attributes):
        tools = [{"type": "function"}]
        attrs = captured_span_attributes(
            invocation_parameters={"tools": tools},
        )
        value = attrs["llm.invocation_parameters.tools"]
        assert isinstance(value, str)
        assert json.loads(value) == tools

    def test_per_key_values_respect_payload_and_redaction_filters(
        self, captured_span_attributes
    ):
        attrs = captured_span_attributes(
            invocation_parameters={
                "api_key": "sk-1234",
                "messages": [{"role": "user", "content": "hi"}],
                "temperature": 0.3,
            },
        )
        per_key = {
            key: value
            for key, value in attrs.items()
            if key.startswith("llm.invocation_parameters.")
        }
        assert set(per_key) == {
            "llm.invocation_parameters.api_key",
            "llm.invocation_parameters.temperature",
        }
        assert per_key["llm.invocation_parameters.api_key"] == "***1234"


class TestOtherAttributesUnaffected:
    def test_model_name_and_messages_still_traced(self, captured_span_attributes):
        attrs = captured_span_attributes(
            input_messages=[{"role": "user", "content": "hi"}],
            invocation_parameters={"temperature": 0.3},
            model_name="gpt-4o-mini",
        )
        assert attrs["llm.model_name"] == "gpt-4o-mini"
        assert any(key.startswith("llm.input_messages.") for key in attrs)
