import pytest
from guardrails import Guard
from guardrails.errors import ValidationError
from typing import Optional
from tests.integration_tests.test_assets.validators import RegexMatch

pytest.importorskip("llama_index")

from llama_index.core.query_engine import BaseQueryEngine  # noqa
from llama_index.core.schema import QueryBundle  # noqa
from llama_index.core.base.response.schema import Response  # noqa
from llama_index.core.prompts.mixin import PromptMixinType  # noqa
from llama_index.core.callbacks import CallbackManager  # noqa


class MockQueryEngine(BaseQueryEngine):
    def __init__(self, callback_manager: Optional[CallbackManager] = None):
        super().__init__(callback_manager)

    def _query(self, query_bundle: QueryBundle) -> Response:
        return Response(response="Mock response")

    async def _aquery(self, query_bundle: QueryBundle) -> Response:
        return Response(response="Mock async query response")

    def _get_prompt_modules(self) -> PromptMixinType:
        return {}


@pytest.fixture
def guard():
    return Guard().use(RegexMatch("Mock response", match_type="search"))


class TestGuardrailsQueryEngine:
    def test_guardrails_engine_init(self, guard):
        from guardrails.integrations.llama_index import GuardrailsQueryEngine

        engine = MockQueryEngine()
        guardrails_engine = GuardrailsQueryEngine(engine, guard)
        assert isinstance(guardrails_engine, GuardrailsQueryEngine)
        assert guardrails_engine.guard == guard

    def test_guardrails_engine_query(self, guard):
        from guardrails.integrations.llama_index import GuardrailsQueryEngine

        engine = MockQueryEngine()
        guardrails_engine = GuardrailsQueryEngine(engine, guard)

        result = guardrails_engine._query(QueryBundle(query_str="Mock response"))
        assert isinstance(result, Response)
        assert result.response == "Mock response"

    def test_guardrails_engine_query_validation_failure(self, guard):
        from guardrails.integrations.llama_index import GuardrailsQueryEngine

        engine = MockQueryEngine()
        guardrails_engine = GuardrailsQueryEngine(engine, guard)

        engine._query = lambda _: Response(response="Invalid response")

        with pytest.raises(ValidationError, match="Validation failed"):
            guardrails_engine._query(QueryBundle(query_str="Invalid query"))
