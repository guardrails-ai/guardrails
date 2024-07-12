from typing import Optional
import io
import sys

import pytest

from guardrails.guard import Guard
from guardrails.validators import ReadingTime, RegexMatch
from guardrails.integrations.langchain.guard_runnable import GuardRunnable
from guardrails.errors import ValidationError


@pytest.fixture
def guard_runnable():
    return GuardRunnable(
        Guard()
        .use(
            RegexMatch("Ice cream", match_type="search", on_fail="refrain"), on="output"
        )
        .use(ReadingTime(0.05, on_fail="refrain"))
    )


@pytest.mark.parametrize(
    "output,throws",
    [
        ("Ice cream is frozen.", False),
        ("Ice cream is a frozen dairy product that is consumed in many places.", True),
        ("This response isn't relevant.", True),
    ],
)
def test_guard_as_runnable(guard_runnable: GuardRunnable, output: str, throws: bool):
    from langchain_core.language_models import LanguageModelInput
    from langchain_core.messages import AIMessage, BaseMessage
    from langchain_core.output_parsers import StrOutputParser
    from langchain_core.prompts import ChatPromptTemplate
    from langchain_core.runnables import Runnable, RunnableConfig

    class MockModel(Runnable):
        def invoke(
            self, input: LanguageModelInput, config: Optional[RunnableConfig] = None
        ) -> BaseMessage:
            return AIMessage(content=output)

    prompt = ChatPromptTemplate.from_template("ELIF: {topic}")
    model = MockModel()
    output_parser = StrOutputParser()

    chain = prompt | model | guard_runnable | output_parser

    topic = "ice cream"
    if throws:
        with pytest.raises(ValidationError) as exc_info:
            chain.invoke({"topic": topic})

        assert str(exc_info.value) == (
            "The response from the LLM failed validation!"
            "See `guard.history` for more details."
        )

        assert guard_runnable.guard.history.last.status == "fail"
        assert guard_runnable.guard.history.last.status == "fail"

    else:
        result = chain.invoke({"topic": topic})

        assert result == output


def test_guard_runnable_with_callback_config(guard_runnable):
    from langchain_core.callbacks import CallbackManager
    from langchain_core.tracers import ConsoleCallbackHandler
    from langchain_core.runnables import RunnableConfig

    console_handler = ConsoleCallbackHandler()
    callback_manager = CallbackManager([console_handler])
    config_with_callbacks = RunnableConfig(callbacks=callback_manager)

    captured_output = io.StringIO()
    sys.stdout = captured_output
    guard_runnable.invoke("Ice cream is sweet", config=config_with_callbacks)
    sys.stdout = sys.__stdout__

    assert "Ice cream is sweet" in captured_output.getvalue()
