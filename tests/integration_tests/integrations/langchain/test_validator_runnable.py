from typing import Optional

import pytest


@pytest.mark.parametrize(
    "output,throws,expected_error",
    [
        ("Ice cream is frozen.", False, None),
        (
            "Ice cream is a frozen dairy product that is consumed in many places.",
            True,
            "String should be readable within 0.05 minutes.",
        ),
        ("This response isn't relevant.", True, "Result must match Ice cream"),
    ],
)
def test_guard_as_runnable(output: str, throws: bool, expected_error: Optional[str]):
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
    regex_match = RegexMatch(
        "Ice cream", match_type="search", on_fail="refrain"
    ).to_runnable()
    reading_time = ReadingTime(0.05, on_fail="refrain").to_runnable()

    output_parser = StrOutputParser()

    chain = prompt | model | regex_match | reading_time | output_parser

    topic = "ice cream"
    if throws:
        with pytest.raises(ValidationError) as exc_info:
            chain.invoke({"topic": topic})

        assert str(exc_info.value) == (
            "The response from the LLM failed validation!" f" {expected_error}"
        )

    else:
        result = chain.invoke({"topic": topic})

        assert result == output
