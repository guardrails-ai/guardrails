from typing import Optional

import pytest

from guardrails.guard import Guard


@pytest.mark.parametrize(
    "output,throws",
    [
        ("Ice cream is frozen.", False),
        ("Ice cream is a frozen dairy product that is consumed in many places.", True),
        ("This response isn't relevant.", True),
    ],
)
def test_guard_as_runnable(output: str, throws: bool):
    from langchain_core.language_models import LanguageModelInput
    from langchain_core.messages import AIMessage, BaseMessage
    from langchain_core.output_parsers import StrOutputParser
    from langchain_core.prompts import ChatPromptTemplate
    from langchain_core.runnables import Runnable, RunnableConfig

    from guardrails.errors import ValidationError
    from tests.integration_tests.test_assets.validators import ReadingTime, RegexMatch

    class MockModel(Runnable):
        def invoke(
            self, input: LanguageModelInput, config: Optional[RunnableConfig] = None
        ) -> BaseMessage:
            return AIMessage(content=output)

    prompt = ChatPromptTemplate.from_template("ELIF: {topic}")
    model = MockModel()
    guard = (
        Guard()
        .use(
            RegexMatch("Ice cream", match_type="search", on_fail="refrain"), on="output"
        )
        .use(ReadingTime(0.05, on_fail="refrain"))  # 3 seconds
    )
    output_parser = StrOutputParser()

    chain = prompt | model | guard.to_runnable() | output_parser

    topic = "ice cream"
    if throws:
        with pytest.raises(ValidationError) as exc_info:
            chain.invoke({"topic": topic})

        assert str(exc_info.value) == (
            "The response from the LLM failed validation!"
            "See `guard.history` for more details."
        )

        assert guard.history.last.status == "fail"
        assert guard.history.last.status == "fail"

    else:
        result = chain.invoke({"topic": topic})

        assert result == output
