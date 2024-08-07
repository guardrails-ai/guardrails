import os
import sys

from opentelemetry import trace
from opentelemetry.trace import Tracer

# TODO: Make the option between GRPC and HTTP configurable
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.resources import SERVICE_NAME, Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import (
    BatchSpanProcessor,
    ConsoleSpanExporter,
    SimpleSpanProcessor,
)

import threading

from guardrails.version import GUARDRAILS_VERSION


class DefaultOtlpTracer:
    _instance = None
    _lock = threading.Lock()
    """Whether to use a local server for running Guardrails."""
    tracer: Tracer

    def __new__(cls, resource_name: str) -> "DefaultOtlpTracer":
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super(DefaultOtlpTracer, cls).__new__(cls)
                    cls._instance._initialize(resource_name)
        return cls._instance

    def _initialize(self, resource_name: str):
        envvars_exist = os.environ.get("OTEL_EXPORTER_OTLP_PROTOCOL") and (
            os.environ.get("OTEL_EXPORTER_OTLP_TRACES_ENDPOINT")
            or os.environ.get("OTEL_EXPORTER_OTLP_ENDPOINT")
        )

        resource = Resource(attributes={SERVICE_NAME: resource_name})

        traceProvider = TracerProvider(resource=resource)
        span_exporter = OTLPSpanExporter()
        if envvars_exist:
            processor = BatchSpanProcessor(span_exporter=span_exporter)
        else:
            processor = SimpleSpanProcessor(ConsoleSpanExporter(out=sys.stderr))

        traceProvider.add_span_processor(processor)
        trace.set_tracer_provider(traceProvider)

        self.tracer = trace.get_tracer("guardrails-ai", GUARDRAILS_VERSION)


def default_otlp_tracer(resource_name: str = "guardrails") -> Tracer:
    """This tracer will emit spans directly to an otlp endpoint, configured by
    the following environment variables:

    OTEL_EXPORTER_OTLP_PROTOCOL
    OTEL_EXPORTER_OTLP_TRACES_ENDPOINT
    OTEL_EXPORTER_OTLP_ENDPOINT
    OTEL_EXPORTER_OTLP_HEADERS

    We recommend using Grafana to collect your metrics. A full example of how to
    do that is in our (docs)[https://docs.guardrails.com/telemetry]
    """
    return DefaultOtlpTracer(resource_name).tracer
