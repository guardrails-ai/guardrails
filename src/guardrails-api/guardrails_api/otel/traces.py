import os
from typing import Optional
from opentelemetry import trace
from opentelemetry.trace import Tracer
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import (
    BatchSpanProcessor,
    SimpleSpanProcessor,
    ConsoleSpanExporter,
    SpanExporter,
    SpanProcessor,
)
from opentelemetry.exporter.otlp.proto.http.trace_exporter import (
    OTLPSpanExporter as HttpSpanExporter,
)
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import (
    OTLPSpanExporter as GrpcSpanExporter,
)
from guardrails_api.otel.constants import none


def traces_are_disabled() -> bool:
    otel_traces_exporter = os.environ.get("OTEL_TRACES_EXPORTER", none)
    return otel_traces_exporter == none


def get_tracer(name: Optional[str] = None) -> Tracer:
    tracer_name = name or os.environ.get("OTEL_SERVICE_NAME", "guardrails-api")
    tracer = trace.get_tracer(tracer_name)

    return tracer


def get_span_exporter(exporter_type: str) -> SpanExporter:
    if exporter_type == "otlp":
        otlp_protocol = os.environ.get("OTEL_EXPORTER_OTLP_PROTOCOL", "http/protobuf")
        trace_exporter = HttpSpanExporter()
        if otlp_protocol == "grpc":
            trace_exporter = GrpcSpanExporter()
        return trace_exporter
    elif exporter_type == "console":
        return ConsoleSpanExporter()


def set_span_processors(
    tracer_provider: TracerProvider,
    exporter: SpanExporter,
    use_batch: bool,
) -> SpanProcessor:
    span_processor = BatchSpanProcessor(exporter)
    if not use_batch:
        span_processor = SimpleSpanProcessor(exporter)
    tracer_provider.add_span_processor(span_processor)


def initialize_tracer():
    if not traces_are_disabled():
        tracer_provider = trace.get_tracer_provider()

        trace_exporter_settings = os.environ.get("OTEL_TRACES_EXPORTER", "none").split(",")
        trace_exporters = [get_span_exporter(e) for e in trace_exporter_settings if e != "none"]

        use_batch = os.environ.get("OTEL_PROCESS_IN_BATCH", "true") == "true"
        for exporter in trace_exporters:
            set_span_processors(tracer_provider, exporter, use_batch)

        # Initialize singleton
        get_tracer()
