import os
from typing import Optional
from opentelemetry import metrics
from opentelemetry.metrics import Meter
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.metrics.export import (
    ConsoleMetricExporter,
    PeriodicExportingMetricReader,
    MetricExporter,
)
from opentelemetry.exporter.otlp.proto.http.metric_exporter import (
    OTLPMetricExporter as HttpMetricExporter,
)
from opentelemetry.exporter.otlp.proto.grpc.metric_exporter import (
    OTLPMetricExporter as GrpcMetricExporter,
)
from guardrails_api.otel.constants import none


def metrics_are_disabled() -> bool:
    otel_metrics_exporter = os.environ.get("OTEL_METRICS_EXPORTER", none)
    return otel_metrics_exporter == none


def get_meter(name: Optional[str] = None) -> Meter:
    meter_name = name or os.environ.get("OTEL_SERVICE_NAME", "guardrails-api")
    meter = metrics.get_meter(meter_name)

    return meter


def get_metrics_exporter(exporter_type: str) -> MetricExporter:
    if exporter_type == "otlp":
        otlp_protocol = os.environ.get("OTEL_EXPORTER_OTLP_PROTOCOL", "http/protobuf")
        metrics_exporter = HttpMetricExporter()
        if otlp_protocol == "grpc":
            metrics_exporter = GrpcMetricExporter()
        return metrics_exporter
    elif exporter_type == "console":
        return ConsoleMetricExporter()


def initialize_metrics_collector():
    if not metrics_are_disabled():
        metrics_exporter_settings = os.environ.get("OTEL_METRICS_EXPORTER", "none").split(",")
        metric_exporters = [
            get_metrics_exporter(e) for e in metrics_exporter_settings if e != "none"
        ]

        metric_readers = []
        for exporter in metric_exporters:
            metric_readers.append(PeriodicExportingMetricReader(exporter))

        provider = MeterProvider(metric_readers=metric_readers)
        metrics.set_meter_provider(provider)

        get_meter()
