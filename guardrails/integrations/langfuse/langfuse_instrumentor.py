"""Sends Guardrails' OpenTelemetry traces to Langfuse.

Guardrails already emits a full trace per guard invocation. This integration
does not create spans of its own; it makes Langfuse export the existing ones,
translates a few attributes into the Langfuse data model, and turns validation
results into scores.
"""

from typing import Any, Dict, Optional

from opentelemetry import trace
from opentelemetry.sdk.trace import ReadableSpan, TracerProvider

try:
    from langfuse import Langfuse
    from langfuse._client.resource_manager import LangfuseResourceManager
    from langfuse.span_filter import is_default_export_span
    from langfuse.types import (
        MaskOtelSpansParams,
        MaskOtelSpansResult,
        OtelSpanPatch,
    )
except ImportError:
    raise ImportError(
        "Please install langfuse to use this instrumentor: "
        'pip install "guardrails-ai[langfuse]"'
    )

from guardrails.integrations.langfuse.attribute_mapping import (
    GUARDRAILS_INSTRUMENTATION_SCOPES,
    is_guardrails_scope,
    map_attributes,
)
from guardrails.integrations.langfuse.score_processor import (
    GuardrailsScoreSpanProcessor,
)
from guardrails.logger import logger
from guardrails.settings import settings


def guardrails_should_export_span(span: ReadableSpan) -> bool:
    """Export filter that admits Guardrails spans alongside Langfuse's defaults.

    Langfuse's default filter keeps only Langfuse spans, spans carrying
    ``gen_ai.*`` attributes, and known LLM instrumentation scopes. Guardrails is
    none of those, so without this every Guardrails span is silently dropped.
    """
    scope = span.instrumentation_scope
    if scope is not None and is_guardrails_scope(scope.name):
        return True
    return is_default_export_span(span)


def guardrails_mask_otel_spans(
    *, params: MaskOtelSpansParams
) -> Optional[MaskOtelSpansResult]:
    """Adds ``langfuse.*`` attributes to Guardrails spans at export time."""
    try:
        patches: Dict[Any, OtelSpanPatch] = {}
        for identifier, span in params.spans.items():
            mapped = map_attributes(span.attributes or {})
            if mapped:
                patches[identifier] = OtelSpanPatch(set_attributes=mapped)
        return MaskOtelSpansResult(span_patches=patches)
    except Exception as e:
        # Raising here would cost the user the entire export batch, so
        # enrichment is best-effort: the spans still ship, just unmapped.
        logger.warning(f"Failed to map Guardrails spans for Langfuse: {e}")
        return None


class LangfuseInstrumentor:
    """Instruments Guardrails to send traces to Langfuse.

    Credentials are read from ``LANGFUSE_PUBLIC_KEY``, ``LANGFUSE_SECRET_KEY``
    and ``LANGFUSE_BASE_URL``.

    ```python
    from guardrails.integrations.langfuse import LangfuseInstrumentor

    langfuse = LangfuseInstrumentor().instrument()
    ```

    This must run before any other Langfuse client is created for the same
    public key: Langfuse caches one resource manager per key and returns the
    existing one on a later call, discarding the export filter and mapping hook
    this integration depends on.

    A single Langfuse project per process is supported.
    """

    def __init__(
        self,
        *,
        emit_validation_scores: bool = True,
        **langfuse_kwargs: Any,
    ):
        self.emit_validation_scores = emit_validation_scores
        self._langfuse_kwargs = langfuse_kwargs
        self._client: Optional[Langfuse] = None
        self._score_processor: Optional[GuardrailsScoreSpanProcessor] = None

    def instrument(self) -> Langfuse:
        """Configure Langfuse to receive Guardrails traces. Idempotent."""
        if self._client is not None:
            return self._client

        self._reject_isolated_tracer_provider()
        self._reject_preexisting_client()

        if settings.disable_tracing:
            logger.warning(
                "Guardrails tracing is disabled (settings.disable_tracing), so no"
                " spans will reach Langfuse. Note that MlFlowInstrumentor sets"
                " this flag."
            )

        self._client = Langfuse(
            should_export_span=guardrails_should_export_span,
            mask_otel_spans=guardrails_mask_otel_spans,
            **self._langfuse_kwargs,
        )

        if self.emit_validation_scores:
            self._attach_score_processor(self._client)

        return self._client

    def flush(self) -> None:
        """Flush pending spans and scores. Needed in short-lived processes."""
        if self._client is not None:
            self._client.flush()

    def _reject_isolated_tracer_provider(self) -> None:
        provider = self._langfuse_kwargs.get("tracer_provider")
        if provider is not None and provider is not trace.get_tracer_provider():
            raise ValueError(
                "LangfuseInstrumentor does not support an isolated tracer_provider."
                " Guardrails takes its tracers from the global TracerProvider, so"
                " an isolated one would receive no Guardrails spans. Omit"
                " tracer_provider, or pass the global one."
            )

    def _reject_preexisting_client(self) -> None:
        instances = getattr(LangfuseResourceManager, "_instances", None)
        if instances is None:
            logger.warning(
                "Could not verify whether a Langfuse client already exists; if one"
                " does, Guardrails spans may not be exported."
            )
            return
        if instances:
            raise RuntimeError(
                "A Langfuse client already exists in this process. Langfuse reuses"
                " the cached client for a public key and ignores the export filter"
                " and mapping hook this integration needs, so Guardrails spans"
                " would be dropped. Call LangfuseInstrumentor().instrument() before"
                " creating any other Langfuse client, and use the client it"
                " returns."
            )

    def _attach_score_processor(self, client: Langfuse) -> None:
        provider = trace.get_tracer_provider()
        if not isinstance(provider, TracerProvider):
            logger.warning(
                "The global TracerProvider is not an OpenTelemetry SDK"
                " TracerProvider, so Guardrails validation scores cannot be"
                " emitted."
            )
            return
        self._score_processor = GuardrailsScoreSpanProcessor(client)
        provider.add_span_processor(self._score_processor)


__all__ = [
    "GUARDRAILS_INSTRUMENTATION_SCOPES",
    "LangfuseInstrumentor",
    "guardrails_mask_otel_spans",
    "guardrails_should_export_span",
]
