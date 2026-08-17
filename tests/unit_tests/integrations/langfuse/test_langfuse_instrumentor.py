from unittest.mock import MagicMock

import pytest
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

try:
    import langfuse
except ImportError:
    langfuse = None

pytestmark = pytest.mark.skipif(langfuse is None, reason="langfuse not installed.")


@pytest.fixture(autouse=True)
def langfuse_env(monkeypatch):
    monkeypatch.setenv("LANGFUSE_PUBLIC_KEY", "pk-lf-test")
    monkeypatch.setenv("LANGFUSE_SECRET_KEY", "sk-lf-test")
    monkeypatch.setenv("LANGFUSE_BASE_URL", "http://localhost:3000")


@pytest.fixture(autouse=True)
def reset_langfuse():
    """Langfuse caches one resource manager per public key; without resetting,
    a client built by one test leaks into the next."""
    from langfuse._client.resource_manager import LangfuseResourceManager

    LangfuseResourceManager.reset()
    trace._TRACER_PROVIDER_SET_ONCE._done = False
    yield
    LangfuseResourceManager.reset()


def make_instrumentor(**kwargs):
    from guardrails.integrations.langfuse import LangfuseInstrumentor

    kwargs.setdefault("span_exporter", InMemorySpanExporter())
    return LangfuseInstrumentor(**kwargs)


def make_span(scope_name, attributes=None):
    span = MagicMock()
    span.instrumentation_scope = (
        MagicMock(name=scope_name) if scope_name is not None else None
    )
    # MagicMock(name=...) sets the mock's repr, not the attribute.
    if scope_name is not None:
        span.instrumentation_scope.name = scope_name
    span.attributes = attributes or {}
    return span


class TestShouldExportSpan:
    """The load-bearing part: Langfuse's default filter drops every Guardrails
    span, so this predicate is what makes the integration work at all."""

    @pytest.mark.parametrize(
        "scope", ["guardrails-ai", "guardrails.telemetry.guard_tracing"]
    )
    def test_guardrails_spans_are_exported(self, scope):
        from guardrails.integrations.langfuse import guardrails_should_export_span

        assert guardrails_should_export_span(make_span(scope)) is True

    def test_defers_to_langfuse_for_other_scopes(self, mocker):
        from guardrails.integrations import langfuse as gr_langfuse
        from guardrails.integrations.langfuse import guardrails_should_export_span

        default = mocker.patch.object(
            gr_langfuse.langfuse_instrumentor,
            "is_default_export_span",
            return_value=False,
        )
        span = make_span("sqlalchemy")

        assert guardrails_should_export_span(span) is False
        default.assert_called_once_with(span)

    def test_missing_scope_defers_to_langfuse(self, mocker):
        from guardrails.integrations import langfuse as gr_langfuse
        from guardrails.integrations.langfuse import guardrails_should_export_span

        mocker.patch.object(
            gr_langfuse.langfuse_instrumentor,
            "is_default_export_span",
            return_value=True,
        )

        assert guardrails_should_export_span(make_span(None)) is True


class TestMaskOtelSpans:
    def test_maps_guardrails_spans_only(self):
        from langfuse.types import MaskOtelSpansParams

        from guardrails.integrations.langfuse import guardrails_mask_otel_spans

        guard_span = make_span("guardrails-ai", {"type": "guardrails/guard"})
        other_span = make_span("openai", {"http.method": "POST"})
        params = MaskOtelSpansParams(spans={"a": guard_span, "b": other_span})

        result = guardrails_mask_otel_spans(params=params)

        assert "a" in result.span_patches
        assert (
            result.span_patches["a"].set_attributes["langfuse.observation.type"]
            == "span"
        )
        assert "b" not in result.span_patches

    def test_returns_none_instead_of_raising(self, mocker):
        """Raising from this hook costs the user the whole export batch, so
        enrichment must be best-effort."""
        from langfuse.types import MaskOtelSpansParams

        from guardrails.integrations import langfuse as gr_langfuse
        from guardrails.integrations.langfuse import guardrails_mask_otel_spans

        mocker.patch.object(
            gr_langfuse.langfuse_instrumentor,
            "map_attributes",
            side_effect=ValueError("boom"),
        )
        params = MaskOtelSpansParams(
            spans={"a": make_span("guardrails-ai", {"type": "guardrails/guard"})}
        )

        assert guardrails_mask_otel_spans(params=params) is None


class TestInstrument:
    def test_is_idempotent(self):
        instrumentor = make_instrumentor()

        first = instrumentor.instrument()
        second = instrumentor.instrument()

        assert first is second

    def test_rejects_preexisting_client(self):
        from langfuse import Langfuse

        Langfuse()

        with pytest.raises(RuntimeError, match="already exists"):
            make_instrumentor().instrument()

    def test_rejects_isolated_tracer_provider(self):
        isolated = TracerProvider()

        with pytest.raises(ValueError, match="isolated tracer_provider"):
            make_instrumentor(tracer_provider=isolated).instrument()

    def test_accepts_the_global_tracer_provider(self):
        provider = TracerProvider()
        trace._TRACER_PROVIDER_SET_ONCE._done = False
        trace.set_tracer_provider(provider)

        instrumentor = make_instrumentor(tracer_provider=trace.get_tracer_provider())

        assert instrumentor.instrument() is not None

    def test_warns_when_guardrails_tracing_is_disabled(self, mocker):
        from guardrails.integrations import langfuse as gr_langfuse
        from guardrails.settings import settings

        warning = mocker.patch.object(
            gr_langfuse.langfuse_instrumentor.logger, "warning"
        )
        mocker.patch.object(settings, "disable_tracing", True)

        make_instrumentor().instrument()

        assert any("disable_tracing" in str(call) for call in warning.call_args_list)

    def test_score_processor_can_be_disabled(self):
        instrumentor = make_instrumentor(emit_validation_scores=False)
        instrumentor.instrument()

        assert instrumentor._score_processor is None

    def test_score_processor_is_attached_by_default(self):
        instrumentor = make_instrumentor()
        instrumentor.instrument()

        assert instrumentor._score_processor is not None
