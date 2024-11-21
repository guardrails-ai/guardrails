# 3 tests
# 1. Test streaming with LiteLLMCallable (mock openai.Completion.create)
# 2. Test streaming with OpenAIChatCallable (mock openai.ChatCompletion.create)
# 3. Test string schema streaming
# Using the LowerCase Validator, and a custom validator to show new streaming behavior
import json
from typing import Any, Callable, Dict, Iterable, List, Optional, Union

import openai
import pytest
from pydantic import BaseModel, Field

import guardrails as gd
from guardrails.utils.casting_utils import to_int
from guardrails.validator_base import (
    ErrorSpan,
    FailResult,
    OnFailAction,
    PassResult,
    ValidationResult,
    Validator,
    register_validator,
)
from tests.integration_tests.test_assets.validators import LowerCase, MockDetectPII

expected_raw_output = {"statement": "I am DOING well, and I HOPE you aRe too."}
expected_fix_output = {"statement": "i am doing well, and i hope you are too."}
expected_noop_output = {"statement": "I am DOING well, and I HOPE you aRe too."}
expected_filter_refrain_output = {}


@register_validator(name="minsentencelength", data_type=["string", "list"])
class MinSentenceLengthValidator(Validator):
    def __init__(
        self,
        min: Optional[int] = None,
        max: Optional[int] = None,
        on_fail: Optional[Callable] = None,
    ):
        super().__init__(
            on_fail=on_fail,
            min=min,
            max=max,
        )
        self._min = to_int(min)
        self._max = to_int(max)

    def sentence_split(self, value):
        return list(map(lambda x: x + ".", value.split(".")[:-1]))

    def validate(self, value: Union[str, List], metadata: Dict) -> ValidationResult:
        sentences = self.sentence_split(value)
        error_spans = []
        index = 0
        for sentence in sentences:
            if len(sentence) < self._min:
                error_spans.append(
                    ErrorSpan(
                        start=index,
                        end=index + len(sentence),
                        reason=f"Sentence has length less than {self._min}. "
                        f"Please return a longer output, "
                        f"that is shorter than {self._max} characters.",
                    )
                )
            if len(sentence) > self._max:
                error_spans.append(
                    ErrorSpan(
                        start=index,
                        end=index + len(sentence),
                        reason=f"Sentence has length greater than {self._max}. "
                        f"Please return a shorter output, "
                        f"that is shorter than {self._max} characters.",
                    )
                )
            index = index + len(sentence)
        if len(error_spans) > 0:
            return FailResult(
                validated_chunk=value,
                error_spans=error_spans,
                error_message=f"Sentence has length less than {self._min}. "
                f"Please return a longer output, "
                f"that is shorter than {self._max} characters.",
            )
        return PassResult(validated_chunk=value)

    def validate_stream(self, chunk: Any, metadata: Dict, **kwargs) -> ValidationResult:
        return super().validate_stream(chunk, metadata, **kwargs)


class Delta:
    content: str

    def __init__(self, content):
        self.content = content


class Choice:
    text: str
    finish_reason: str
    index: int
    delta: Delta

    def __init__(self, text, delta, finish_reason, index=0):
        self.index = index
        self.delta = delta
        self.text = text
        self.finish_reason = finish_reason


class MockOpenAIV1ChunkResponse:
    choices: list
    model: str

    def __init__(self, choices, model):
        self.choices = choices
        self.model = model


def mock_openai_completion_create(chunks):
    # Returns a generator
    def gen():
        for chunk in chunks:
            yield MockOpenAIV1ChunkResponse(
                choices=[
                    Choice(
                        text=chunk,
                        delta=None,
                        finish_reason=None,
                    )
                ],
                model="OpenAI model name",
            )

    return gen()


def mock_openai_chat_completion_create(chunks):
    # Returns a generator
    def gen():
        for chunk in chunks:
            yield MockOpenAIV1ChunkResponse(
                choices=[
                    Choice(
                        text=None,
                        delta=Delta(content=chunk),
                        finish_reason=None,
                    )
                ],
                model="OpenAI model name",
            )

    return gen()


class LowerCaseFix(BaseModel):
    statement: str = Field(
        description="Validates whether the text is in lower case.",
        validators=[LowerCase(on_fail=OnFailAction.FIX)],
    )


class LowerCaseNoop(BaseModel):
    statement: str = Field(
        description="Validates whether the text is in lower case.",
        validators=[LowerCase(on_fail=OnFailAction.NOOP)],
    )


class LowerCaseFilter(BaseModel):
    statement: str = Field(
        description="Validates whether the text is in lower case.",
        validators=[LowerCase(on_fail=OnFailAction.FILTER)],
    )


class LowerCaseRefrain(BaseModel):
    statement: str = Field(
        description="Validates whether the text is in lower case.",
        validators=[LowerCase(on_fail=OnFailAction.REFRAIN)],
    )


expected_minsentence_noop_output = ""


class MinSentenceLengthNoOp(BaseModel):
    statement: str = Field(
        description="Validates whether the text is in lower case.",
        validators=[MinSentenceLengthValidator(on_fail=OnFailAction.NOOP)],
    )


STR_PROMPT = "Say something nice to me."

PROMPT = """
Say something nice to me.

${gr.complete_json_suffix}
"""

JSON_LLM_CHUNKS = [
    '{"statement":',
    ' "I am DOING',
    " well, and I",
    " HOPE you aRe",
    ' too."}',
]

MESSAGES = [
    {
        "role": "user",
        "content": PROMPT,
    }
]

STR_MESSAGES = [
    {
        "role": "user",
        "content": STR_PROMPT,
    }
]


@pytest.mark.parametrize(
    "guard, expected_validated_output",
    [
        (
            gd.Guard.for_pydantic(output_class=LowerCaseNoop, messages=MESSAGES),
            expected_noop_output,
        ),
        (
            gd.Guard.for_pydantic(output_class=LowerCaseFix, messages=MESSAGES),
            expected_fix_output,
        ),
        (
            gd.Guard.for_pydantic(output_class=LowerCaseFilter, messages=MESSAGES),
            expected_filter_refrain_output,
        ),
        (
            gd.Guard.for_pydantic(output_class=LowerCaseRefrain, messages=MESSAGES),
            expected_filter_refrain_output,
        ),
    ],
)
def test_streaming_with_openai_callable(
    mocker,
    guard,
    expected_validated_output,
):
    """Test streaming with LiteLLMCallable.

    Mocks openai.Completion.create.
    """
    mocker.patch(
        "openai.resources.Completions.create",
        return_value=mock_openai_completion_create(JSON_LLM_CHUNKS),
    )

    method = openai.completions.create

    method.__name__ = "mock_openai_completion_create"

    generator = guard(
        method,
        engine="text-davinci-003",
        max_tokens=10,
        temperature=0,
        stream=True,
    )

    assert isinstance(generator, Iterable)

    for op in generator:
        actual_output = op

    assert actual_output.raw_llm_output == json.dumps(expected_raw_output)
    assert actual_output.validated_output == expected_validated_output


@pytest.mark.parametrize(
    "guard, expected_validated_output",
    [
        (
            gd.Guard.for_pydantic(output_class=LowerCaseNoop, messages=MESSAGES),
            expected_noop_output,
        ),
        (
            gd.Guard.for_pydantic(output_class=LowerCaseFix, messages=MESSAGES),
            expected_fix_output,
        ),
        (
            gd.Guard.for_pydantic(output_class=LowerCaseFilter, messages=MESSAGES),
            expected_filter_refrain_output,
        ),
        (
            gd.Guard.for_pydantic(output_class=LowerCaseRefrain, messages=MESSAGES),
            expected_filter_refrain_output,
        ),
    ],
)
def test_streaming_with_openai_chat_callable(
    mocker,
    guard,
    expected_validated_output,
):
    """Test streaming with OpenAIChatCallable.

    Mocks openai.ChatCompletion.create.
    """
    mocker.patch(
        "openai.resources.chat.completions.Completions.create",
        return_value=mock_openai_chat_completion_create(JSON_LLM_CHUNKS),
    )

    method = openai.chat.completions.create

    method.__name__ = "mock_openai_chat_completion_create"

    generator = guard(
        method,
        model="gpt-3.5-turbo",
        max_tokens=10,
        temperature=0,
        stream=True,
    )

    assert isinstance(generator, Iterable)

    actual_output = ""
    for op in generator:
        actual_output = op

    assert actual_output.raw_llm_output == json.dumps(expected_raw_output)
    assert actual_output.validated_output == expected_validated_output


STR_LLM_CHUNKS = [
    # 38 characters
    "This sentence is simply just ",
    "too long.",
    # 25 characters long
    "This ",
    "sentence ",
    "is 2 ",
    "short."
    # 29 characters long
    "This sentence is just ",
    "right.",
]


@pytest.mark.parametrize(
    "guard, expected_error_spans",
    [
        (
            gd.Guard.for_string(
                # only the middle sentence should pass
                validators=[
                    MinSentenceLengthValidator(26, 30, on_fail=OnFailAction.NOOP)
                ],
                messages=STR_MESSAGES,
            ),
            # each value is a tuple
            # first is expected text inside span
            # second is the reason for failure
            [
                [
                    "This sentence is simply just too long.",
                    (
                        "Sentence has length greater than 30. "
                        "Please return a shorter output, "
                        "that is shorter than 30 characters."
                    ),
                ],
                [
                    "This sentence is 2 short.",
                    (
                        "Sentence has length less than 26. "
                        "Please return a longer output, "
                        "that is shorter than 30 characters."
                    ),
                ],
            ],
        )
    ],
)
def test_string_schema_streaming_with_openai_chat(mocker, guard, expected_error_spans):
    """Test string schema streaming with OpenAIChatCallable.

    Mocks openai.ChatCompletion.create.
    """
    mocker.patch(
        "openai.resources.chat.completions.Completions.create",
        return_value=mock_openai_chat_completion_create(STR_LLM_CHUNKS),
    )

    method = openai.chat.completions.create

    method.__name__ = "mock_openai_chat_completion_create"
    generator = guard(
        method,
        model="gpt-3.5-turbo",
        max_tokens=10,
        temperature=0,
        stream=True,
    )

    assert isinstance(generator, Iterable)

    accumulated_output = ""
    for op in generator:
        accumulated_output += op.raw_llm_output
    error_spans = guard.error_spans_in_output()

    assert len(error_spans) == len(expected_error_spans)
    for error_span, expected in zip(error_spans, expected_error_spans):
        assert accumulated_output[error_span.start : error_span.end] == expected[0]
        assert error_span.reason == expected[1]
    # TODO assert something about these error spans


POETRY_CHUNKS = [
    '"John, under ',
    "GOLDEN bridges",
    ", roams,\n",
    "SAN Francisco's ",
    "hills, his HOME.\n",
    "Dreams of",
    " FOG, and salty AIR,\n",
    "In his HEART",
    ", he's always THERE.",
]


def test_noop_behavior_two_validators(mocker):
    mocker.patch(
        "openai.resources.chat.completions.Completions.create",
        return_value=mock_openai_chat_completion_create(POETRY_CHUNKS),
    )

    guard = gd.Guard().use_many(
        MockDetectPII(
            on_fail=OnFailAction.NOOP,
            pii_entities="pii",
            replace_map={"John": "<PERSON>", "SAN Francisco's": "<LOCATION>"},
        ),
        LowerCase(on_fail=OnFailAction.NOOP),
    )
    prompt = """Write me a 4 line poem about John in San Francisco. 
    Make every third word all caps."""
    gen = guard(
        llm_api=openai.chat.completions.create,
        messages=[
            {
                "role": "user",
                "content": prompt,
            }
        ],
        model="gpt-4",
        stream=True,
    )
    text = ""
    original = ""
    for res in gen:
        original = original + res.raw_llm_output
        text = text + res.validated_output
    assert (
        text
        == """"John, under GOLDEN bridges, roams,
SAN Francisco's hills, his HOME.
Dreams of FOG, and salty AIR,
In his HEART, he's always THERE."""
    )
    assert (
        original
        == """"John, under GOLDEN bridges, roams,
SAN Francisco's hills, his HOME.
Dreams of FOG, and salty AIR,
In his HEART, he's always THERE."""
    )


def test_fix_behavior_one_validator(mocker):
    mocker.patch(
        "openai.resources.chat.completions.Completions.create",
        return_value=mock_openai_chat_completion_create(POETRY_CHUNKS),
    )

    guard = gd.Guard().use_many(
        LowerCase(on_fail=OnFailAction.FIX),
    )
    prompt = """Write me a 4 line poem about John in San Francisco. 
    Make every third word all caps."""
    gen = guard(
        llm_api=openai.chat.completions.create,
        messages=[
            {
                "role": "user",
                "content": prompt,
            }
        ],
        model="gpt-4",
        stream=True,
    )
    text = ""
    original = ""
    for res in gen:
        original = original + res.raw_llm_output
        text = text + res.validated_output
    assert (
        text
        == """"john, under golden bridges, roams,
san francisco's hills, his home.dreams of fog, and salty air,
in his heart, he's always there."""
    )
    assert (
        original
        == """"John, under GOLDEN bridges, roams,
SAN Francisco's hills, his HOME.
Dreams of FOG, and salty AIR,
In his HEART, he's always THERE."""
    )


def test_fix_behavior_two_validators(mocker):
    mocker.patch(
        "openai.resources.chat.completions.Completions.create",
        return_value=mock_openai_chat_completion_create(POETRY_CHUNKS),
    )

    guard = gd.Guard().use_many(
        MockDetectPII(
            on_fail=OnFailAction.FIX,
            pii_entities="pii",
            replace_map={"John": "<PERSON>", "SAN Francisco's": "<LOCATION>"},
        ),
        LowerCase(on_fail=OnFailAction.FIX),
    )
    prompt = """Write me a 4 line poem about John in San Francisco. 
    Make every third word all caps."""
    gen = guard(
        llm_api=openai.chat.completions.create,
        messages=[
            {
                "role": "user",
                "content": prompt,
            }
        ],
        model="gpt-4",
        stream=True,
    )
    text = ""
    original = ""
    for res in gen:
        original = original + res.raw_llm_output
        text = text + res.validated_output
    assert (
        text
        == """"<PERSON>, under golden bridges, roams,
<LOCATION> hills, his home.dreams of fog, and salty air,
in his heart, he's always there."""
    )
    assert (
        original
        == """"John, under GOLDEN bridges, roams,
SAN Francisco's hills, his HOME.
Dreams of FOG, and salty AIR,
In his HEART, he's always THERE."""
    )


def test_fix_behavior_three_validators(mocker):
    mocker.patch(
        "openai.resources.chat.completions.Completions.create",
        return_value=mock_openai_chat_completion_create(POETRY_CHUNKS),
    )

    guard = gd.Guard().use_many(
        MockDetectPII(
            on_fail=OnFailAction.FIX,
            pii_entities="pii",
            replace_map={"John": "<PERSON>", "SAN Francisco's": "<LOCATION>"},
        ),
        LowerCase(on_fail=OnFailAction.FIX),
        # UpperCase(on_fail=OnFailAction.FIX),
        MockDetectPII(
            on_fail=OnFailAction.FIX,
            pii_entities="pii",
            replace_map={
                "John": "REDACTED!!",
                "SAN Francisco's": "REDACTED!!",
                "GOLDEN": "purple!!",
            },
        ),
    )
    prompt = """Write me a 4 line poem about John in San Francisco. 
    Make every third word all caps."""
    gen = guard(
        llm_api=openai.chat.completions.create,
        messages=[
            {
                "role": "user",
                "content": prompt,
            }
        ],
        model="gpt-4",
        stream=True,
    )
    text = ""
    original = ""
    for res in gen:
        original = original + res.raw_llm_output
        text = text + res.validated_output
    print("FINAL TEXT", text)
    assert (
        text
        == """"REDACTED!!, under purple!! bridges, roams,
<LOCATION> hills, his home.dreams of fog, and salty air,
in his heart, he's always there."""
    )
    assert (
        original
        == """"John, under GOLDEN bridges, roams,
SAN Francisco's hills, his HOME.
Dreams of FOG, and salty AIR,
In his HEART, he's always THERE."""
    )


# This case does not work!
# def test_fix_behavior_three_validators_overlap(mocker):
#     mocker.patch(
#         "openai.resources.chat.completions.Completions.create",
#         return_value=mock_openai_chat_completion_create(POETRY_CHUNKS),
#     )

#     guard = gd.Guard().use_many(
#         MockDetectPII(
#             on_fail=OnFailAction.FIX,
#             pii_entities="pii",
#             replace_map={"John": "<PERSON>", "SAN Francisco's": "<LOCATION>"},
#         ),
#         LowerCase(on_fail=OnFailAction.FIX),
#         # UpperCase(on_fail=OnFailAction.FIX),
#         MockDetectPII(
#             on_fail=OnFailAction.FIX,
#             pii_entities="pii",
#             replace_map={
#                 "John, under GOLDEN": "REDACTED!!",
#                 "SAN Francisco's hills": "REDACTED!!",
#                 "GOLDEN bridges": "gold!!!!",
#             },
#         ),
#     )
#     prompt = """Write me a 4 line poem about John in San Francisco.
#     Make every third word all caps."""
#     gen = guard(
#         llm_api=openai.chat.completions.create,
#         messages=MESSAGES,
#         model="gpt-4",
#         stream=True,
#     )
#     text = ""
#     original = ""
#     for res in gen:
#         original = original + res.raw_llm_output
#         text = text + res.validated_output
#     print("TEXT", text)
#     assert (
#         text
#         == """"REDACTED!!, under gold!!!! bridges, roams,
# <LOCATION> hills, his home.
# dreams of fog, and salty air,
# in his heart, he's always there."""
#     )
#     assert (
#         original
#         == """"John, under GOLDEN bridges, roams,
# SAN Francisco's hills, his HOME.
# Dreams of FOG, and salty AIR,
# In his HEART, he's always THERE."""
#     )


def test_refrain_behavior(mocker):
    mocker.patch(
        "openai.resources.chat.completions.Completions.create",
        return_value=mock_openai_chat_completion_create(POETRY_CHUNKS),
    )

    guard = gd.Guard().use_many(
        MockDetectPII(
            on_fail=OnFailAction.REFRAIN,
            pii_entities="pii",
            replace_map={"John": "<PERSON>", "SAN Francisco's": "<LOCATION>"},
        ),
        LowerCase(on_fail=OnFailAction.FIX),
    )

    prompt = """Write me a 4 line poem about John in San Francisco. 
    Make every third word all caps."""
    gen = guard(
        llm_api=openai.chat.completions.create,
        messages=[
            {
                "role": "user",
                "content": prompt,
            }
        ],
        model="gpt-4",
        stream=True,
    )
    text = ""
    original = ""
    for res in gen:
        original = original + res.raw_llm_output
        text = text + res.validated_output
    assert text == ""
    assert (
        original
        == """"John, under GOLDEN bridges, roams,
SAN Francisco's hills, his HOME.
"""
    )


def test_filter_behavior(mocker):
    mocker.patch(
        "openai.resources.chat.completions.Completions.create",
        return_value=mock_openai_chat_completion_create(POETRY_CHUNKS),
    )

    guard = gd.Guard().use_many(
        MockDetectPII(
            on_fail=OnFailAction.FIX,
            pii_entities="pii",
            replace_map={"John": "<PERSON>", "SAN Francisco's": "<LOCATION>"},
        ),
        LowerCase(on_fail=OnFailAction.FILTER),
    )
    prompt = """Write me a 4 line poem about John in San Francisco. 
    Make every third word all caps."""
    gen = guard(
        llm_api=openai.chat.completions.create,
        messages=[
            {
                "role": "user",
                "content": prompt,
            }
        ],
        model="gpt-4",
        stream=True,
    )
    text = ""
    original = ""
    for res in gen:
        original = original + res.raw_llm_output
        text = text + res.validated_output
    assert text == ""
    assert (
        original
        == """"John, under GOLDEN bridges, roams,
SAN Francisco's hills, his HOME.
"""
    )
