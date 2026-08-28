import pytest
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from opentelemetry.sdk.trace.export import SimpleSpanProcessor


@pytest.fixture(autouse=True)
def reset_singletons():
    from guardrails.telemetry.default_otel_collector_tracer_mod import (
        DefaultOtelCollectorTracer,
    )
    from guardrails.telemetry.default_otlp_tracer_mod import DefaultOtlpTracer
    from opentelemetry import trace

    DefaultOtelCollectorTracer._instance = None
    DefaultOtlpTracer._instance = None
    trace._TRACER_PROVIDER_SET_ONCE._done = False


class TestTelemetry:
    def test_private_traces_go_to_user_telem_sink(self, mocker):
        private_exporter = InMemorySpanExporter()
        mocker.patch(
            "guardrails.telemetry.default_otel_collector_tracer_mod.OTLPSpanExporter",
            return_value=private_exporter,
        )
        mocker.patch(
            "guardrails.telemetry.default_otel_collector_tracer_mod.BatchSpanProcessor",
            return_value=SimpleSpanProcessor(private_exporter),
        )

        from guardrails.telemetry import default_otel_collector_tracer
        from guardrails import Guard
        from tests.integration_tests.test_assets.validators import LowerCase

        default_otel_collector_tracer()

        guard = Guard(name="integration-test-guard").use(LowerCase())

        guard.parse("hello world")

        private_spans = private_exporter.get_finished_spans()

        assert len(private_spans) == 4

        for span in private_spans:
            assert span.name in ["guard", "step", "call", "lower-case.validate"]
            assert span.attributes.get("type") in [
                "guardrails/guard",
                "guardrails/guard/step",
                "guardrails/guard/step/call",
                "guardrails/guard/step/validator",
            ]

    def test_no_cross_contamination(self, mocker):
        private_exporter = InMemorySpanExporter()
        mocker.patch(
            "guardrails.telemetry.default_otel_collector_tracer_mod.OTLPSpanExporter",
            return_value=private_exporter,
        )
        mocker.patch(
            "guardrails.telemetry.default_otel_collector_tracer_mod.BatchSpanProcessor",
            return_value=SimpleSpanProcessor(private_exporter),
        )

        from guardrails.telemetry import default_otel_collector_tracer
        from guardrails import Guard
        from tests.integration_tests.test_assets.validators import LowerCase

        default_otel_collector_tracer()

        guard = Guard(name="integration-test-guard").use(LowerCase())

        guard.parse("hello world")

        private_spans = private_exporter.get_finished_spans()

        assert len(private_spans) == 4
        for span in private_spans:
            assert span.name in ["guard", "step", "call", "lower-case.validate"]
            assert span.attributes.get("type") in [
                "guardrails/guard",
                "guardrails/guard/step",
                "guardrails/guard/step/call",
                "guardrails/guard/step/validator",
            ]
