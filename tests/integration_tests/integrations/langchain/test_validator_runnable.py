from typing import Any, Optional
import io
import sys


import pytest

from guardrails.errors import ValidationError
from tests.integration_tests.test_assets.validators import ReadingTime, RegexMatch


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
def test_validator_runnable(output: str, throws: bool, expected_error: Optional[str]):
    from langchain_core.language_models import LanguageModelInput
    from langchain_core.messages import AIMessage, BaseMessage
    from langchain_core.output_parsers import StrOutputParser
    from langchain_core.prompts import ChatPromptTemplate
    from langchain_core.runnables import Runnable, RunnableConfig

    class MockModel(Runnable):
        def invoke(
            self,
            input: LanguageModelInput,
            config: Optional[RunnableConfig] = None,
            **kwargs: Any,
        ) -> BaseMessage:
            return AIMessage(content=output)

    prompt = ChatPromptTemplate.from_template("ELIF: {topic}")
    model = MockModel()
    regex_match = RegexMatch("Ice cream", match_type="search", on_fail="refrain").to_runnable()
    reading_time = ReadingTime(0.05, on_fail="refrain").to_runnable()

    output_parser = StrOutputParser()

    chain = prompt | model | regex_match | reading_time | output_parser

    topic = "ice cream"
    if throws:
        with pytest.raises(ValidationError) as exc_info:
            chain.invoke({"topic": topic})

        assert str(exc_info.value) == (
            f"The response from the LLM failed validation! {expected_error}"
        )

    else:
        result = chain.invoke({"topic": topic})

        assert result == output


def test_validator_runnable_with_callback_config():
    from langchain_core.callbacks import CallbackManager
    from langchain_core.tracers import ConsoleCallbackHandler
    from langchain_core.runnables import RunnableConfig

    console_handler = ConsoleCallbackHandler()
    callback_manager = CallbackManager([console_handler])
    config_with_callbacks = RunnableConfig(callbacks=callback_manager)

    regex_match = RegexMatch("Ice cream", match_type="search", on_fail="exception").to_runnable()

    captured_output = io.StringIO()
    sys.stdout = captured_output

    result = regex_match.invoke("Ice cream is delicious.", config=config_with_callbacks)
    assert result == "Ice cream is delicious."

    with pytest.raises(ValidationError) as exc_info:
        regex_match.invoke("Chocolate is delicious.", config=config_with_callbacks)

    assert "The response from the LLM failed validation!" in str(exc_info.value)
    assert "Result must match Ice cream" in str(exc_info.value)

    sys.stdout = sys.__stdout__

    console_output = captured_output.getvalue()
    assert "Ice cream is delicious." in console_output
    assert "Chocolate is delicious." in console_output
