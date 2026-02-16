from typing import Any, Optional
import io
import sys

import pytest
from pydantic import PrivateAttr

from guardrails.guard import Guard
from guardrails.integrations.langchain.guard_runnable import GuardRunnable
from guardrails.errors import ValidationError
from guardrails.classes import ValidationOutcome

from tests.integration_tests.test_assets.validators import ReadingTime, RegexMatch


@pytest.fixture
def guard_runnable():
    return GuardRunnable(
        Guard()
        .use(
            RegexMatch("Ice cream", match_type="search", on_fail="refrain"), on="output"
        )
        .use(ReadingTime(0.05, on_fail="noop"))
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
            self,
            input: LanguageModelInput,
            config: Optional[RunnableConfig] = None,
            **kwargs: Any,
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


@pytest.mark.parametrize(
    "succeed_on_attempt, max_retries, expected_attempts, expected_result",
    [
        (1, 2, 1, "Succeeded on attempt 1"),
        (2, 2, 1, "Failed attempt 1"),
        (2, None, 1, "Failed attempt 1"),
        (2, 0, 1, "Failed attempt 1"),
    ],
)
def test_guard_runnable_max_retries(
    succeed_on_attempt, max_retries, expected_attempts, expected_result
):
    from langchain_core.runnables import RunnableConfig

    class CountingGuard(Guard):
        _attempt_count: int = PrivateAttr(default=0)
        _succeed_on_attempt: int = PrivateAttr()

        def __init__(self, succeed_on_attempt: int, **kwargs):
            super().__init__(**kwargs)
            self._succeed_on_attempt = succeed_on_attempt

        def validate(self, value):
            self._attempt_count += 1
            if self._attempt_count >= self._succeed_on_attempt:
                return ValidationOutcome(
                    call_id="0",  # type: ignore
                    raw_llm_output=value,
                    validated_output=f"Succeeded on attempt {self._attempt_count}",
                    validation_passed=True,
                )
            raise ValidationError(f"Failed attempt {self._attempt_count}")

        @property
        def attempt_count(self):
            return self._attempt_count

    guard = CountingGuard(succeed_on_attempt)
    runnable = GuardRunnable(guard)

    config = (
        RunnableConfig(max_retries=max_retries) if max_retries is not None else None
    )

    if "Failed" in expected_result:
        with pytest.raises(ValidationError) as exc_info:
            runnable.invoke("test input", config=config)
        assert expected_result in str(exc_info.value)
    else:
        result = runnable.invoke("test input", config=config)
        assert result == expected_result

    assert guard.attempt_count == expected_attempts
