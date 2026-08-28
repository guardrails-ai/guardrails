"""End-to-end checks that Guardrails spans survive Langfuse's export filter and
arrive correctly mapped.

Assertions run against the spans an ``InMemorySpanExporter`` receives *behind*
the real ``LangfuseSpanProcessor``, so the default export filter and the
export-stage mapping hook are both exercised rather than bypassed.
"""

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
from opentelemetry import trace
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
    from langfuse._client.resource_manager import LangfuseResourceManager

    # reset() shuts each instance down before clearing, so batch processors and
    # their threads do not leak between tests.
    LangfuseResourceManager.reset()
    trace._TRACER_PROVIDER_SET_ONCE._done = False
    yield
    LangfuseResourceManager.reset()


@pytest.fixture
def instrumented():
    """An instrumented Langfuse client with an in-memory sink and mocked scores."""
    from guardrails.integrations.langfuse import LangfuseInstrumentor

    exporter = InMemorySpanExporter()
    instrumentor = LangfuseInstrumentor(span_exporter=exporter, flush_at=1)
    client = instrumentor.instrument()
    client.create_score = MagicMock()
    return SimpleNamespace(instrumentor=instrumentor, client=client, exporter=exporter)


def spans_by_name(exporter):
    return {span.name: span for span in exporter.get_finished_spans()}


def observation_types(exporter):
    return {
        span.name: (span.attributes or {}).get("langfuse.observation.type")
        for span in exporter.get_finished_spans()
    }


def litellm_response(model="gpt-4o-mini-2024-07-18", content="hello world"):
    """Minimal stand-in for a litellm ModelResponse."""
    return SimpleNamespace(
        model=model,
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(
                    content=content, role="assistant", to_dict=lambda: {}
                )
            )
        ],
        usage=SimpleNamespace(completion_tokens=7, prompt_tokens=11),
    )


class TestNonStreaming:
    def test_spans_survive_the_filter_and_are_mapped(self, instrumented, mocker):
        import litellm

        mocker.patch.object(litellm, "completion", return_value=litellm_response())

        from guardrails import Guard
        from tests.integration_tests.test_assets.validators import LowerCase

        guard = Guard(name="langfuse-guard").use(LowerCase(on_fail="noop"))
        guard(model="gpt-4o-mini", messages=[{"role": "user", "content": "hi"}])
        instrumented.instrumentor.flush()

        types = observation_types(instrumented.exporter)

        assert types == {
            "guard": "span",
            "step": "span",
            "call": "generation",
            "lower-case.validate": "guardrail",
        }

    def test_generation_carries_model_and_usage(self, instrumented, mocker):
        import litellm

        mocker.patch.object(litellm, "completion", return_value=litellm_response())

        from guardrails import Guard

        guard = Guard(name="langfuse-usage-guard")
        guard(model="gpt-4o-mini", messages=[{"role": "user", "content": "hi"}])
        instrumented.instrumentor.flush()

        call = spans_by_name(instrumented.exporter)["call"]
        attributes = call.attributes or {}

        # The served model, not the requested one -- cost is priced on it.
        assert attributes["llm.model_name"] == "gpt-4o-mini-2024-07-18"
        assert attributes["llm.token_count.prompt"] == 11
        assert attributes["llm.token_count.completion"] == 7
        assert attributes["llm.token_count.total"] == 18

    def test_trace_is_named_after_the_guard(self, instrumented, mocker):
        import litellm

        mocker.patch.object(litellm, "completion", return_value=litellm_response())

        from guardrails import Guard

        guard = Guard(name="named-guard")
        guard(model="gpt-4o-mini", messages=[{"role": "user", "content": "hi"}])
        instrumented.instrumentor.flush()

        guard_span = spans_by_name(instrumented.exporter)["guard"]

        assert (guard_span.attributes or {})["langfuse.trace.name"] == "named-guard"


class TestStreaming:
    """Streaming traces are structurally complete, but usage and cost are absent:
    Guardrails' providers return the stream before final-usage telemetry runs.
    These tests deliberately assert no usage."""

    def test_sync_streaming_call_is_a_generation(self, instrumented):
        from guardrails import Guard

        guard = Guard(name="sync-stream-guard")
        list(
            guard(
                lambda *args, **kwargs: iter(["hello", " world"]),
                messages=[{"role": "user", "content": "hi"}],
                stream=True,
            )
        )
        instrumented.instrumentor.flush()

        call = spans_by_name(instrumented.exporter)["call"]

        assert (call.attributes or {}).get("langfuse.observation.type") == "generation"
        assert "llm.token_count.total" not in (call.attributes or {})

    @pytest.mark.asyncio
    async def test_async_streaming_call_is_a_generation(self, instrumented):
        from guardrails import AsyncGuard

        async def astream(*args, **kwargs):
            # AsyncArbitraryCallable awaits the callable and then reads
            # `.completion_stream`, mirroring litellm's CustomStreamWrapper.
            async def chunks():
                for chunk in ["hello", " world"]:
                    yield chunk

            return SimpleNamespace(completion_stream=chunks())

        guard = AsyncGuard(name="async-stream-guard")
        async for _ in await guard(
            astream,
            messages=[{"role": "user", "content": "hi"}],
            stream=True,
        ):
            pass
        instrumented.instrumentor.flush()

        call = spans_by_name(instrumented.exporter)["call"]

        assert (call.attributes or {}).get("langfuse.observation.type") == "generation"


class TestScores:
    def test_failing_validator_scores_zero_with_the_error(self, instrumented):
        from guardrails import Guard
        from tests.integration_tests.test_assets.validators import LowerCase

        guard = Guard(name="scored-guard").use(LowerCase(on_fail="noop"))
        guard(
            lambda *args, **kwargs: "HELLO WORLD",
            messages=[{"role": "user", "content": "hi"}],
        )
        instrumented.instrumentor.flush()

        scores = {
            call.kwargs["name"]: call.kwargs
            for call in instrumented.client.create_score.call_args_list
        }

        assert scores["guardrails.lower-case"]["value"] == 0.0
        assert scores["guardrails.lower-case"]["data_type"] == "BOOLEAN"
        assert "not lower case" in scores["guardrails.lower-case"]["comment"]
        assert scores["guardrails.validation_passed"]["value"] == 0.0

    def test_passing_validator_scores_one(self, instrumented):
        from guardrails import Guard
        from tests.integration_tests.test_assets.validators import LowerCase

        guard = Guard(name="passing-guard").use(LowerCase(on_fail="noop"))
        guard(
            lambda *args, **kwargs: "hello world",
            messages=[{"role": "user", "content": "hi"}],
        )
        instrumented.instrumentor.flush()

        scores = {
            call.kwargs["name"]: call.kwargs
            for call in instrumented.client.create_score.call_args_list
        }

        assert scores["guardrails.lower-case"]["value"] == 1.0
        assert scores["guardrails.lower-case"]["comment"] is None
        assert scores["guardrails.validation_passed"]["value"] == 1.0

    def test_scores_reference_the_span_they_came_from(self, instrumented):
        from opentelemetry.trace import format_span_id, format_trace_id

        from guardrails import Guard
        from tests.integration_tests.test_assets.validators import LowerCase

        guard = Guard(name="linked-guard").use(LowerCase(on_fail="noop"))
        guard(
            lambda *args, **kwargs: "hello world",
            messages=[{"role": "user", "content": "hi"}],
        )
        instrumented.instrumentor.flush()

        validator_span = spans_by_name(instrumented.exporter)["lower-case.validate"]
        context = validator_span.get_span_context()
        score = next(
            call.kwargs
            for call in instrumented.client.create_score.call_args_list
            if call.kwargs["name"] == "guardrails.lower-case"
        )

        assert score["trace_id"] == format_trace_id(context.trace_id)
        assert score["observation_id"] == format_span_id(context.span_id)
