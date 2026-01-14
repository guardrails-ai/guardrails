from opentelemetry import trace
from opentelemetry.trace import Tracer

# TODO: Make the option between GRPC and HTTP configurable
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.resources import SERVICE_NAME, Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor

import threading

from guardrails.version import GUARDRAILS_VERSION


class DefaultOtelCollectorTracer:
    _instance = None
    _lock = threading.Lock()
    """Whether to use a local server for running Guardrails."""
    tracer: Tracer

    def __new__(cls, resource_name: str) -> "DefaultOtelCollectorTracer":
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super(DefaultOtelCollectorTracer, cls).__new__(cls)
                    cls._instance._initialize(resource_name)
        return cls._instance

    def _initialize(self, resource_name: str):
        resource = Resource(attributes={SERVICE_NAME: resource_name})

        traceProvider = TracerProvider(resource=resource)
        processor = BatchSpanProcessor(OTLPSpanExporter())
        traceProvider.add_span_processor(processor)
        trace.set_tracer_provider(traceProvider)

        self.tracer = traceProvider.get_tracer("guardrails-ai", GUARDRAILS_VERSION)


def default_otel_collector_tracer(resource_name: str = "guardrails") -> Tracer:
    """This is the standard otel tracer set to talk to a grpc open telemetry
    collector running on port 4317."""
    return DefaultOtelCollectorTracer(resource_name).tracer
