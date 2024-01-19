# Imports
from opentelemetry import trace

# 2 exporters available: HTTP and GRPC, only use one at a time
from opentelemetry.exporter.otlp.proto.http.trace_exporter import (
    OTLPSpanExporter,
)  # HTTP

# from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import (
#     OTLPSpanExporter,
# )  # GRPC

from opentelemetry.sdk.resources import SERVICE_NAME, Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import (
    BatchSpanProcessor,
    ConsoleSpanExporter,
    SimpleSpanProcessor,
)
from opentelemetry.trace.propagation.tracecontext import TraceContextTextMapPropagator


class HubTelemetry:
    """Singleton class for initializing a tracer for Guardrails Hub"""

    _instance = None
    _service_name = None
    _endpoint = None
    _tracer_name = None
    _resource = None
    _tracer_provider = None
    _processor = None
    _tracer = None
    _prop = None
    _carrier = {}

    def __new__(
        cls,
        service_name: str = "guardrails-hub",
        endpoint: str = "http://localhost:4318/v1/traces",  # HTTP: 4318, GRPC: 4317
        tracer_name: str = "gr_hub",
        export_locally: bool = False,
    ):
        if cls._instance is None:
            print("Creating HubTelemetry instance...")
            cls._instance = super(HubTelemetry, cls).__new__(cls)
            print("Initializing HubTelemetry instance...")
            cls._instance.initialize_tracer(
                service_name, endpoint, tracer_name, export_locally
            )
        return cls._instance

    def initialize_tracer(
        self,
        service_name: str,
        endpoint: str,
        tracer_name: str,
        export_locally: bool,
    ):
        """Initializes a tracer for Guardrails Hub"""

        self._service_name = service_name
        self._endpoint = endpoint
        self._tracer_name = tracer_name

        # Create a resource
        # Service name is required for most backends
        self._resource = Resource(attributes={SERVICE_NAME: self._service_name})

        # Create a tracer provider and a processor
        self._tracer_provider = TracerProvider(resource=self._resource)

        if export_locally:
            self._processor = SimpleSpanProcessor(ConsoleSpanExporter())
        else:
            self._processor = SimpleSpanProcessor(
                OTLPSpanExporter(endpoint=self._endpoint)
            )

        # Add the processor to the provider
        self._tracer_provider.add_span_processor(self._processor)

        # Set the tracer provider and return a tracer
        trace.set_tracer_provider(self._tracer_provider)
        self._tracer = trace.get_tracer(self._tracer_name)

        self._prop = TraceContextTextMapPropagator()

    def get_tracer(self):
        """Returns the tracer"""

        return self._tracer

    def inject_current_context(self) -> None:
        """Injects the current context into the carrier"""
        self._prop.inject(carrier=self._carrier)

    def extract_current_context(self):
        """Extracts the current context from the carrier"""

        context = self._prop.extract(carrier=self._carrier)
        return context
