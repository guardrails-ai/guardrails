import typing
from typing import Optional

import typer
from opentelemetry import metrics, trace
from opentelemetry.context import Context
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.trace.export import ConsoleSpanExporter
from opentelemetry.sdk.metrics.export import ConsoleMetricExporter
from opentelemetry.sdk.resources import SERVICE_NAME, SERVICE_VERSION, Resource
from opentelemetry.sdk.trace import TracerProvider, Span, ReadableSpan
from opentelemetry.sdk.trace.export import BatchSpanProcessor

#from guardrails.cli.guardrails import guardrails
#from guardrails.cli.logger import logger
#from guardrails.utils.telemetry_utils import get_tracer, get_current_context


@guardrails.command()
def watch(
    refresh_interval: Optional[float] = typer.Option(
        default=0.5,
        help="How frequently (in seconds) should the telemetry be refreshed?",
    ),
    endpoint: Optional[str] = typer.Option(
      default="http://localhost:4317",
      help="The "
    ),
    port: Optional[int] = typer.Option(
        default=8000,
        help="The port to run the server on.",
    )
):
    otlp_exporter = OTLPSpanExporter(endpoint="http://localhost:4317", insecure=True)
    print(refresh_interval)
    print(endpoint)
    print(port)
