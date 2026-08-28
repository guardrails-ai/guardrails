"""Emits Langfuse scores from finished Guardrails validator and guard spans."""

from typing import Any, Optional

from opentelemetry.sdk.trace import ReadableSpan, SpanProcessor
from opentelemetry.trace import format_span_id, format_trace_id

from guardrails.integrations.langfuse.attribute_mapping import (
    GUARD,
    VALIDATOR,
    guard_validation_passed,
    validator_failed,
)
from guardrails.logger import logger

GUARD_SCORE_NAME = "guardrails.validation_passed"


class GuardrailsScoreSpanProcessor(SpanProcessor):
    """Turns validation results into Langfuse scores.

    Scores, unlike span metadata, can be charted over time -- which is the point
    of sending guard runs to Langfuse: pass rate per validator.
    """

    def __init__(self, client: Any):
        self._client = client

    def on_end(self, span: ReadableSpan) -> None:
        # Runs on the exporting thread; must never raise into it.
        try:
            self._score(span)
        except Exception as e:  # pragma: no cover - defensive
            logger.debug(f"Failed to emit Langfuse score for span {span.name}: {e}")

    def _score(self, span: ReadableSpan) -> None:
        attributes = dict(span.attributes or {})
        span_type = attributes.get("type")
        context = span.get_span_context()

        if context is None or not context.is_valid:
            return

        if span_type == VALIDATOR:
            name = attributes.get("validator.name")
            if not name:
                return
            failed = validator_failed(attributes)
            self._create_score(
                name=f"guardrails.{name}",
                value=0.0 if failed else 1.0,
                context=context,
                comment=(
                    str(attributes.get("validator.validate.output.error_message"))
                    if failed
                    else None
                ),
            )
        elif span_type == GUARD:
            passed = guard_validation_passed(attributes)
            if passed is None:
                return
            self._create_score(
                name=GUARD_SCORE_NAME,
                value=1.0 if passed else 0.0,
                context=context,
            )

    def _create_score(
        self,
        *,
        name: str,
        value: float,
        context: Any,
        comment: Optional[str] = None,
    ) -> None:
        self._client.create_score(
            name=name,
            value=value,
            data_type="BOOLEAN",
            trace_id=format_trace_id(context.trace_id),
            observation_id=format_span_id(context.span_id),
            comment=comment,
        )
