# Imports
import logging
from typing import Optional

from guardrails.settings import settings
from opentelemetry.exporter.otlp.proto.http.trace_exporter import (  # HTTP Exporter
    OTLPSpanExporter,
)
from opentelemetry.sdk.resources import SERVICE_NAME, Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import ConsoleSpanExporter, SimpleSpanProcessor
from opentelemetry.trace.propagation.tracecontext import TraceContextTextMapPropagator
from opentelemetry.trace.propagation import set_span_in_context


class HubTelemetry:
    """Singleton class for initializing a tracer for Guardrails Hub."""

    _instance = None
    _service_name = None
    _endpoint = None
    _tracer_name = None
    _resource = None
    _tracer_provider = None
    _processor = None
    _tracer = None
    _prop = None
    _enabled = False

    def __new__(
        cls,
        service_name: str = "guardrails-hub",
        tracer_name: str = "gr_hub",
        export_locally: bool = False,
        *,
        enabled: Optional[bool] = None,
    ):
        if cls._instance is None:
            logging.debug("Creating HubTelemetry instance...")
            cls._instance = super(HubTelemetry, cls).__new__(cls)
            logging.debug("Initializing HubTelemetry instance...")
            cls._instance.initialize_tracer(
                service_name, tracer_name, export_locally, enabled=enabled
            )
        else:
            logging.debug("Returning existing HubTelemetry instance...")
        return cls._instance

    def initialize_tracer(
        self,
        service_name: str,
        tracer_name: str,
        export_locally: bool,
        *,
        enabled: Optional[bool] = None,
    ):
        """Initializes a tracer for Guardrails Hub."""
        if enabled is None:
            enabled = settings.rc.enable_metrics or False
        self._enabled = enabled
        self._carrier = {}
        self._service_name = service_name
        # self._endpoint = "http://localhost:5318/v1/traces"
        self._endpoint = (
            "https://hty0gc1ok3.execute-api.us-east-1.amazonaws.com/v1/traces"
        )
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
        self._tracer = self._tracer_provider.get_tracer(self._tracer_name)

        self._prop = TraceContextTextMapPropagator()

    def inject_current_context(self, context=None) -> None:
        """Injects the current context into the carrier."""
        if not self._prop:
            return
        if context is not None:
            self._prop.inject(carrier=self._carrier, context=context)
        else:
            self._prop.inject(carrier=self._carrier)

    def extract_current_context(self):
        """Extracts the current context from the carrier."""
        if not self._prop:
            return None
        context = self._prop.extract(carrier=self._carrier)
        return context

    def create_new_span(
        self,
        span_name: str,
        attributes: list,
        # todo deprecate these in 060
        is_parent: bool,  #
        has_parent: bool,  # no-qa
    ):
        """Creates a new span within the tracer with the given name and
        attributes.

        If it's a parent span, the current context is injected into the carrier.
        If it has a parent span, the current context is extracted from the carrier.
        Both the conditions can co-exist e.g. a span can be a parent span which
        also has a parent span.

        Args:
            span_name (str): The name of the span.
            attributes (list): A list of attributes to set on the span.
            is_parent (bool): True if the span is a parent span.
            has_parent (bool): True if the span has a parent span.
        """
        if self._tracer is None:
            return
        with self._tracer.start_span(
            span_name,  # type: ignore (Fails in Python 3.9 for invalid reason)
            context=self.extract_current_context(),
        ) as span:
            context = set_span_in_context(span)
            self.inject_current_context(context=context)

            for attribute in attributes:
                span.set_attribute(attribute[0], attribute[1])
