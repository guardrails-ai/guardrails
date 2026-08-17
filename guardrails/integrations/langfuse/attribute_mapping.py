"""Translation of Guardrails span attributes into the Langfuse data model.

Deliberately free of any ``langfuse`` import so the mapping can be unit tested
without the SDK installed.

Only attributes Langfuse does *not* already understand are mapped here. Langfuse
natively reads the OpenInference attributes Guardrails emits -- ``input.value`` /
``output.value`` become the observation input/output, ``llm.model_name`` becomes
the model, ``llm.token_count.*`` becomes usage, and span status becomes the
level -- so re-mapping those would be dead code.
"""

from typing import Any, Dict, Mapping

GUARD = "guardrails/guard"
STEP = "guardrails/guard/step"
CALL = "guardrails/guard/step/call"
VALIDATOR = "guardrails/guard/step/validator"

#: Guardrails' own tracer scopes. Langfuse's default export filter drops any span
#: that is not a Langfuse span, does not carry ``gen_ai.*`` attributes, and does
#: not come from a known LLM instrumentation scope -- which is all of ours.
GUARDRAILS_INSTRUMENTATION_SCOPES = (
    "guardrails-ai",
    "guardrails.telemetry.guard_tracing",
)

_OBSERVATION_TYPE = "langfuse.observation.type"
_OBSERVATION_LEVEL = "langfuse.observation.level"
_TRACE_NAME = "langfuse.trace.name"
_TRACE_TAGS = "langfuse.trace.tags"
_TRACE_METADATA = "langfuse.trace.metadata"

#: Tag every guard run so Guardrails traffic is filterable in the Langfuse UI
#: when Guardrails is one component among many.
_GUARDRAILS_TAG = "guardrails"

#: Guard span attributes promoted to top-level trace metadata. Langfuse only
#: supports filtering on top-level metadata keys; anything left unmapped lands in
#: a non-filterable ``metadata.attributes`` catch-all.
_PROMOTED_GUARD_ATTRIBUTES = (
    "validation_passed",
    "number_of_reasks",
    "number_of_llm_calls",
    "execution_id",
)


def is_guardrails_scope(scope_name: Any) -> bool:
    return scope_name in GUARDRAILS_INSTRUMENTATION_SCOPES


def validator_failed(attributes: Mapping[str, Any]) -> bool:
    """Whether a validator span records a failed validation.

    Returns False when the outcome is missing rather than guessing.
    """
    return attributes.get("validator.validate.output.outcome") == "fail"


def guard_validation_passed(attributes: Mapping[str, Any]) -> Any:
    """The guard span's validation result, or None when not recorded."""
    return attributes.get("validation_passed")


def map_attributes(attributes: Mapping[str, Any]) -> Dict[str, Any]:
    """Langfuse attributes to add to a Guardrails span.

    Returns an empty dict for spans this integration does not recognise, which
    the caller treats as "leave the span unchanged".
    """
    span_type = attributes.get("type")

    if span_type == GUARD:
        return _map_guard(attributes)
    if span_type == STEP:
        return {_OBSERVATION_TYPE: "span"}
    if span_type == CALL:
        # Typed by what the span *is*, not by whether a model happened to be
        # recorded -- streaming and model-less providers are still generations.
        return {_OBSERVATION_TYPE: "generation"}
    if span_type == VALIDATOR:
        return _map_validator(attributes)
    return {}


def _map_guard(attributes: Mapping[str, Any]) -> Dict[str, Any]:
    mapped: Dict[str, Any] = {
        _OBSERVATION_TYPE: "span",
        _TRACE_TAGS: [_GUARDRAILS_TAG],
    }

    guard_name = attributes.get("guard.name")
    if guard_name:
        # Every Guardrails trace's root span is named "guard", so without this
        # every trace looks identical in the Langfuse trace list.
        mapped[_TRACE_NAME] = str(guard_name)

    for key in _PROMOTED_GUARD_ATTRIBUTES:
        value = attributes.get(key)
        if value is not None:
            mapped[f"{_TRACE_METADATA}.{key}"] = str(value)

    # No level on the guard span even when validation fails: with on_fail="fix"
    # or "reask" a failure is routine, and flagging every root span would make
    # the level meaningless. The failing validator span carries the level, and
    # failed runs stay findable via validation_passed metadata and the score.
    return mapped


def _map_validator(attributes: Mapping[str, Any]) -> Dict[str, Any]:
    mapped: Dict[str, Any] = {_OBSERVATION_TYPE: "guardrail"}
    if validator_failed(attributes):
        mapped[_OBSERVATION_LEVEL] = "WARNING"
    return mapped
