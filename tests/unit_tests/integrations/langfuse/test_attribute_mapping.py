import pytest

from guardrails.integrations.langfuse.attribute_mapping import (
    CALL,
    GUARD,
    STEP,
    VALIDATOR,
    guard_validation_passed,
    is_guardrails_scope,
    map_attributes,
    validator_failed,
)


class TestMapAttributes:
    def test_unknown_span_is_left_unchanged(self):
        assert map_attributes({"type": "something/else"}) == {}
        assert map_attributes({}) == {}

    def test_step_is_a_span(self):
        assert map_attributes({"type": STEP}) == {"langfuse.observation.type": "span"}

    @pytest.mark.parametrize(
        "attributes",
        [
            {"type": CALL},
            {"type": CALL, "llm.model_name": "gpt-4o-mini"},
        ],
        ids=["without_model", "with_model"],
    )
    def test_call_is_a_generation_regardless_of_model(self, attributes):
        """Streaming and model-less providers are still generations, so the type
        must come from the Guardrails span type rather than model presence."""
        assert map_attributes(attributes) == {"langfuse.observation.type": "generation"}

    def test_guard_promotes_filterable_metadata(self):
        mapped = map_attributes(
            {
                "type": GUARD,
                "guard.name": "my-guard",
                "validation_passed": True,
                "number_of_reasks": 2,
                "number_of_llm_calls": 3,
                "execution_id": "abc123",
            }
        )

        assert mapped == {
            "langfuse.observation.type": "span",
            "langfuse.trace.tags": ["guardrails"],
            "langfuse.trace.name": "my-guard",
            "langfuse.trace.metadata.validation_passed": "True",
            "langfuse.trace.metadata.number_of_reasks": "2",
            "langfuse.trace.metadata.number_of_llm_calls": "3",
            "langfuse.trace.metadata.execution_id": "abc123",
        }

    def test_guard_omits_absent_metadata(self):
        mapped = map_attributes({"type": GUARD, "guard.name": "my-guard"})

        assert mapped == {
            "langfuse.observation.type": "span",
            "langfuse.trace.tags": ["guardrails"],
            "langfuse.trace.name": "my-guard",
        }

    @pytest.mark.parametrize("passed", [True, False])
    def test_guard_never_carries_a_level(self, passed):
        """A failure under on_fail="fix"/"reask" is routine, so flagging the root
        span would make the level meaningless. The validator span carries it."""
        mapped = map_attributes({"type": GUARD, "validation_passed": passed})

        assert "langfuse.observation.level" not in mapped

    def test_failed_guard_is_still_findable(self):
        mapped = map_attributes({"type": GUARD, "validation_passed": False})

        assert mapped["langfuse.trace.metadata.validation_passed"] == "False"

    def test_failed_validator_is_flagged(self):
        mapped = map_attributes(
            {"type": VALIDATOR, "validator.validate.output.outcome": "fail"}
        )

        assert mapped == {
            "langfuse.observation.type": "guardrail",
            "langfuse.observation.level": "WARNING",
        }

    def test_passing_validator_has_no_level(self):
        mapped = map_attributes(
            {"type": VALIDATOR, "validator.validate.output.outcome": "pass"}
        )

        assert mapped == {"langfuse.observation.type": "guardrail"}


class TestOutcomeHelpers:
    def test_validator_failed(self):
        assert validator_failed({"validator.validate.output.outcome": "fail"}) is True
        assert validator_failed({"validator.validate.output.outcome": "pass"}) is False

    def test_missing_outcome_is_not_a_failure(self):
        """A missing outcome is unknown, not failed -- guessing would emit a
        wrong score."""
        assert validator_failed({}) is False

    def test_guard_validation_passed(self):
        assert guard_validation_passed({"validation_passed": False}) is False
        assert guard_validation_passed({}) is None


class TestScopes:
    @pytest.mark.parametrize(
        "scope", ["guardrails-ai", "guardrails.telemetry.guard_tracing"]
    )
    def test_guardrails_scopes_are_recognised(self, scope):
        assert is_guardrails_scope(scope) is True

    def test_other_scopes_are_not(self):
        assert is_guardrails_scope("openai") is False
        assert is_guardrails_scope(None) is False
