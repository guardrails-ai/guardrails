import json
import re
from typing import Any, Dict, List
import pytest
from pydantic import BaseModel, Field

from guardrails import Guard, Validator, register_validator
from guardrails.async_guard import AsyncGuard
from guardrails.errors import ValidationError
from guardrails.actions.reask import FieldReAsk
from guardrails.actions.refrain import Refrain
from guardrails.actions.filter import Filter
from guardrails.classes.validation.validation_result import (
    FailResult,
    PassResult,
    ValidationResult,
)
from guardrails.types import OnFailAction
from tests.integration_tests.test_assets.validators import (
    TwoWords,
    ValidLength,
)


@register_validator("mycustomhellovalidator", data_type="string")
def hello_validator(value: Any, metadata: Dict[str, Any]) -> ValidationResult:
    if "hello" in value.lower():
        return FailResult(
            error_message="Hello is too basic, try something more creative.",
            fix_value="hullo",
        )
    return PassResult()


def test_validator_as_tuple():
    # (Callable, on_fail) tuple fix
    class MyModel(BaseModel):
        a_field: str = Field(..., validators=[(hello_validator(), OnFailAction.FIX)])

    guard = Guard.for_pydantic(MyModel)
    output = guard.parse(
        '{"a_field": "hello there yo"}',
        num_reasks=0,
    )

    assert output.validated_output == {"a_field": "hullo"}

    # (string, on_fail) tuple fix

    class MyModel(BaseModel):
        a_field: str = Field(
            ...,
            validators=[
                ("two_words", OnFailAction.REASK),
                ("mycustomhellovalidator", OnFailAction.FIX),
            ],
        )

    guard = Guard.for_pydantic(MyModel)
    output = guard.parse(
        '{"a_field": "hello there yo"}',
        num_reasks=0,
    )

    assert output.validated_output == {"a_field": "hullo"}

    # (Validator, on_fail) tuple fix

    class MyModel(BaseModel):
        a_field: str = Field(..., validators=[(TwoWords(), OnFailAction.FIX)])

    guard = Guard.for_pydantic(MyModel)
    output = guard.parse(
        '{"a_field": "hello there yo"}',
        num_reasks=0,
    )

    assert output.validated_output == {"a_field": "hello there"}

    # (Validator, on_fail) tuple reask

    hullo_reask = FieldReAsk(
        incorrect_value="hello there yo",
        fail_results=[
            FailResult(
                error_message="Hello is too basic, try something more creative.",
                fix_value="hullo",
            )
        ],
        path=["a_field"],
    )

    class MyModel(BaseModel):
        a_field: str = Field(..., validators=[(hello_validator(), OnFailAction.REASK)])

    guard = Guard.for_pydantic(MyModel)

    output = guard.parse(
        '{"a_field": "hello there yo"}',
        num_reasks=0,
    )

    assert output.validated_output == {"a_field": "hullo"}
    assert guard.history.first.iterations.first.reasks[0] == hullo_reask

    hello_reask = FieldReAsk(
        incorrect_value="hello there yo",
        fail_results=[
            FailResult(
                error_message="must be exactly two words",
                fix_value="hello there",
            )
        ],
        path=["a_field"],
    )

    # (string, on_fail) tuple reask

    class MyModel(BaseModel):
        a_field: str = Field(..., validators=[("two-words", OnFailAction.REASK)])

    guard = Guard.for_pydantic(MyModel)

    output = guard.parse(
        '{"a_field": "hello there yo"}',
        num_reasks=0,
    )

    assert output.validated_output == {"a_field": "hello there"}
    assert guard.history.first.iterations.first.reasks[0] == hello_reask

    # (Validator, on_fail) tuple reask

    class MyModel(BaseModel):
        a_field: str = Field(..., validators=[(TwoWords(), OnFailAction.REASK)])

    guard = Guard.for_pydantic(MyModel)

    output = guard.parse(
        '{"a_field": "hello there yo"}',
        num_reasks=0,
    )

    assert output.validated_output == {"a_field": "hello there"}
    assert guard.history.first.iterations.first.reasks[0] == hello_reask

    class MyModel(BaseModel):
        a_field: str = Field(..., validators=["two-words"])

    # Unintentionally supported, but supported nonetheless
    # with pytest.raises(ValueError):
    guard = Guard.for_pydantic(MyModel)
    assert len(guard._validators) == 1


def test_custom_func_validator():
    rail_str = """
    <rail version="0.1">
    <output>
        <string name="greeting"
                validators="mycustomhellovalidator"
                on-fail-mycustomhellovalidator="fix"/>
    </output>
    </rail>
    """

    guard = Guard.for_rail_string(rail_str)

    output = guard.parse(
        '{"greeting": "hello"}',
        num_reasks=0,
    )
    assert output.validated_output == {"greeting": "hullo"}

    call = guard.history.first
    assert call.iterations.length == 1
    validator_log = call.iterations.first.validator_logs[0]
    assert validator_log.validator_name == "mycustomhellovalidator"
    assert validator_log.validation_result == FailResult(
        error_message="Hello is too basic, try something more creative.",
        fix_value="hullo",
    )


def test_bad_validator():
    with pytest.raises(ValueError):

        @register_validator("mycustombadvalidator", data_type="string")
        def validate(value: Any) -> ValidationResult:
            pass


@pytest.mark.parametrize(
    "min,max,expected_xml",
    [
        (0, 12, "length: 0 12"),
        ("0", "12", "length: 0 12"),
        (None, 12, "length: None 12"),
        (1, None, "length: 1 None"),
    ],
)
def test_to_xml_attrib(min, max, expected_xml):
    validator = ValidLength(min=min, max=max)
    xml_validator = validator.to_xml_attrib()

    assert xml_validator == expected_xml


def custom_deprecated_on_fail_handler(value: Any, fail_results: List[FailResult]):
    return value + " deprecated"


def custom_fix_on_fail_handler(value: Any, fail_result: FailResult):
    return value + " " + value


def custom_reask_on_fail_handler(value: Any, fail_result: FailResult):
    return FieldReAsk(incorrect_value=value, fail_results=[fail_result])


def custom_exception_on_fail_handler(value: Any, fail_result: FailResult):
    raise ValidationError("Something went wrong!")


def custom_filter_on_fail_handler(value: Any, fail_result: FailResult):
    return Filter()


def custom_refrain_on_fail_handler(value: Any, fail_result: FailResult):
    return Refrain()


class TestCustomOnFailHandler:
    def test_deprecated_on_fail_handler(self):
        prompt = """
            What kind of pet should I get and what should I name it?

            ${gr.complete_json_suffix_v2}
        """
        messages = [
            {
                "role": "user",
                "content": prompt,
            }
        ]
        output = """
        {
        "pet_type": "dog",
        "name": "Fido"
        }
        """
        expected_result = {"pet_type": "dog deprecated", "name": "Fido"}

        with pytest.warns(
            DeprecationWarning,
            match=re.escape(  # Becuase of square brackets in the message
                "Specifying a List[FailResult] as the second argument"
                " for a custom on_fail handler is deprecated. "
                "Please use FailResult instead."
            ),
        ):
            validator: Validator = TwoWords(on_fail=custom_deprecated_on_fail_handler)  # type: ignore

        class Pet(BaseModel):
            pet_type: str = Field(description="Species of pet", validators=[validator])
            name: str = Field(description="a unique pet name")

        guard = Guard.for_pydantic(output_class=Pet, messages=messages)

        response = guard.parse(output, num_reasks=0)
        assert response.validation_passed is True
        assert response.validated_output == expected_result

    def test_custom_fix(self):
        prompt = """
            What kind of pet should I get and what should I name it?

            ${gr.complete_json_suffix_v2}
        """
        messages = [
            {
                "role": "user",
                "content": prompt,
            }
        ]
        output = """
        {
        "pet_type": "dog",
        "name": "Fido"
        }
        """
        expected_result = {"pet_type": "dog dog", "name": "Fido"}

        validator: Validator = TwoWords(on_fail=custom_fix_on_fail_handler)

        class Pet(BaseModel):
            pet_type: str = Field(description="Species of pet", validators=[validator])
            name: str = Field(description="a unique pet name")

        guard = Guard.for_pydantic(output_class=Pet, messages=messages)

        response = guard.parse(output, num_reasks=0)
        assert response.validation_passed is True
        assert response.validated_output == expected_result

    def test_custom_reask(self):
        prompt = """
            What kind of pet should I get and what should I name it?

            ${gr.complete_json_suffix_v2}
        """
        messages = [
            {
                "role": "user",
                "content": prompt,
            }
        ]
        output = """
        {
        "pet_type": "dog",
        "name": "Fido"
        }
        """
        expected_result = FieldReAsk(
            incorrect_value="dog",
            path=["pet_type"],
            fail_results=[
                FailResult(
                    error_message="must be exactly two words",
                    fix_value="dog dog",
                )
            ],
        )

        validator: Validator = TwoWords(on_fail=custom_reask_on_fail_handler)

        class Pet(BaseModel):
            pet_type: str = Field(description="Species of pet", validators=[validator])
            name: str = Field(description="a unique pet name")

        guard = Guard.for_pydantic(output_class=Pet, messages=messages)

        response = guard.parse(output, num_reasks=0)

        # Why? Because we have a bad habit of applying every fix value
        #       to the output even if the user doesn't ask us to.
        assert response.validation_passed is True
        assert guard.history.first.iterations.first.reasks[0] == expected_result

    def test_custom_exception(self):
        prompt = """
            What kind of pet should I get and what should I name it?

            ${gr.complete_json_suffix_v2}
        """
        messages = [
            {
                "role": "user",
                "content": prompt,
            }
        ]
        output = """
        {
        "pet_type": "dog",
        "name": "Fido"
        }
        """

        validator: Validator = TwoWords(on_fail=custom_exception_on_fail_handler)

        class Pet(BaseModel):
            pet_type: str = Field(description="Species of pet", validators=[validator])
            name: str = Field(description="a unique pet name")

        guard = Guard.for_pydantic(output_class=Pet, messages=messages)

        with pytest.raises(ValidationError) as excinfo:
            guard.parse(output, num_reasks=0)
        assert str(excinfo.value) == "Something went wrong!"

    def test_custom_filter(self):
        prompt = """
            What kind of pet should I get and what should I name it?

            ${gr.complete_json_suffix_v2}
        """
        messages = [
            {
                "role": "user",
                "content": prompt,
            }
        ]
        output = """
        {
        "pet_type": "dog",
        "name": "Fido"
        }
        """

        validator: Validator = TwoWords(on_fail=custom_filter_on_fail_handler)

        class Pet(BaseModel):
            pet_type: str = Field(description="Species of pet", validators=[validator])
            name: str = Field(description="a unique pet name")

        guard = Guard.for_pydantic(output_class=Pet, messages=messages)

        response = guard.parse(output, num_reasks=0)

        # NOTE: This doesn't seem right.
        #       Shouldn't pass if filtering is successful on the target property?
        assert response.validation_passed is False
        assert response.validated_output is None

    def test_custom_refrain(self):
        prompt = """
            What kind of pet should I get and what should I name it?

            ${gr.complete_json_suffix_v2}
        """
        messages = [
            {
                "role": "user",
                "content": prompt,
            }
        ]
        output = """
        {
        "pet_type": "dog",
        "name": "Fido"
        }
        """

        validator: Validator = TwoWords(on_fail=custom_refrain_on_fail_handler)

        class Pet(BaseModel):
            pet_type: str = Field(description="Species of pet", validators=[validator])
            name: str = Field(description="a unique pet name")

        guard = Guard.for_pydantic(output_class=Pet, messages=messages)

        response = guard.parse(output, num_reasks=0)

        assert response.validation_passed is False
        assert response.validated_output is None


class Pet(BaseModel):
    name: str = Field(description="a unique pet name")


def test_input_validation_fix(mocker):
    def mock_llm_api(messages, *args, **kwargs):
        return json.dumps({"name": "Fluffy"})

    # fix returns an amended value for prompt/instructions validation,
    guard = Guard.for_pydantic(output_class=Pet)
    guard.use(TwoWords(on_fail=OnFailAction.FIX), on="messages")

    guard(
        mock_llm_api,
        messages=[
            {
                "role": "user",
                "content": "What kind of pet should I get?",
            }
        ],
    )

    assert guard.history.first.iterations.first.outputs.validation_response == "What kind"

    # but raises for messages validation
    guard = Guard.for_pydantic(output_class=Pet)
    guard.use(TwoWords(on_fail=OnFailAction.EXCEPTION), on="messages")

    with pytest.raises(ValidationError) as excinfo:
        guard(
            mock_llm_api,
            messages=[
                {
                    "role": "user",
                    "content": "What kind of pet should I get?",
                }
            ],
        )
    assert str(excinfo.value) == (
        "Validation failed for field with errors: must be exactly two words"
    )
    assert isinstance(guard.history.first.exception, ValidationError)
    assert guard.history.first.exception == excinfo.value

    # rail messages validation
    guard = Guard.for_rail_string(
        """
<rail version="0.1">
    <messages
        validators="two-words"
        on-fail-two-words="fix"
    >
        <message role="user">This is not two words</message>
        <message role="user">This also is not two words</message>
    </messages>
    <output type="string">
    </output>
</rail>
"""
    )
    guard(
        mock_llm_api,
    )
    assert guard.history.first.iterations.first.outputs.validation_response == "This also"


@pytest.mark.asyncio
async def test_async_messages_validation_fix(mocker):
    async def mock_llm_api(messages, *args, **kwargs) -> str:
        return json.dumps({"name": "Fluffy"})

    # fix returns an amended value for messages validation,
    guard = AsyncGuard.for_pydantic(output_class=Pet)
    guard.use(TwoWords(on_fail=OnFailAction.FIX), on="messages")

    await guard(
        mock_llm_api,
        messages=[
            {
                "role": "user",
                "content": "What kind of pet should I get?",
            }
        ],
    )
    assert guard.history.first.iterations.first.outputs.validation_response == "What kind"

    guard = AsyncGuard.for_pydantic(output_class=Pet)
    guard.use(TwoWords(on_fail=OnFailAction.FIX), on="messages")

    await guard(
        mock_llm_api,
        messages=[
            {
                "role": "user",
                "content": "But really, what kind of pet should I get?",
            }
        ],
    )
    assert guard.history.first.iterations.first.outputs.validation_response == "But really,"

    # but raises for messages validation
    guard = AsyncGuard.for_pydantic(output_class=Pet)
    guard.use(TwoWords(on_fail=OnFailAction.FIX), on="messages")

    await guard(
        mock_llm_api,
        messages=[
            {
                "role": "user",
                "content": "What kind of pet should I get?",
            }
        ],
    )
    first_iter = guard.history.first.iterations.first
    assert first_iter.outputs.validation_response == "What kind"

    # rail prompt validation
    guard = AsyncGuard.for_rail_string(
        """
<rail version="0.1">
<messages
    validators="two-words"
    on-fail-two-words="fix"
>
<message role="user">This is not two words</message>
</messages>
<output type="string">
</output>
</rail>
"""
    )
    await guard(
        mock_llm_api,
    )
    assert guard.history.first.iterations.first.outputs.validation_response == "This is"


@pytest.mark.parametrize(
    "on_fail,structured_messages_error,unstructured_messages_error,",
    [
        (
            OnFailAction.REASK,
            "Messages validation failed: incorrect_value='What kind of pet should I get?' fail_results=[FailResult(outcome='fail', error_message='must be exactly two words', fix_value='What kind', error_spans=None, metadata=None, validated_chunk=None)] additional_properties={} path=None",  # noqa
            "Messages validation failed: incorrect_value='What kind of pet should I get?' fail_results=[FailResult(outcome='fail', error_message='must be exactly two words', fix_value='What kind', error_spans=None, metadata=None, validated_chunk=None)] additional_properties={} path=None",  # noqa
        ),
        (
            OnFailAction.FILTER,
            "Messages validation failed",
            "Messages validation failed",
        ),
        (
            OnFailAction.REFRAIN,
            "Messages validation failed",
            "Messages validation failed",
        ),
        (
            OnFailAction.EXCEPTION,
            "Validation failed for field with errors: must be exactly two words",
            "Validation failed for field with errors: must be exactly two words",
        ),
    ],
)
def test_input_validation_fail(
    on_fail,
    structured_messages_error,
    unstructured_messages_error,
):
    # With Prompt Validation
    guard = Guard.for_pydantic(output_class=Pet)
    guard.use(TwoWords(on_fail=on_fail), on="messages")

    def custom_llm(messages, *args, **kwargs):
        raise Exception(
            "LLM was called when it should not have been!"
            "Input Validation did not raise as expected!"
        )

    # With messages Validation
    guard = Guard.for_pydantic(output_class=Pet)
    guard.use(TwoWords(on_fail=on_fail), on="messages")

    with pytest.raises(ValidationError) as excinfo:
        guard(
            custom_llm,
            messages=[
                {
                    "role": "user",
                    "content": "What kind of pet should I get?",
                }
            ],
        )
    assert str(excinfo.value) == structured_messages_error
    assert isinstance(guard.history.last.exception, ValidationError)
    assert guard.history.last.exception == excinfo.value

    # Rail Prompt Validation
    guard = Guard.for_rail_string(
        f"""
<rail version="0.1">
<messages
    validators="two-words"
    on-fail-two-words="{on_fail.value}"
>
<message role="user">What kind of pet should I get?</message>
</messages>
<output type="string">
</output>
</rail>
"""
    )
    with pytest.raises(ValidationError) as excinfo:
        guard(
            custom_llm,
        )
    assert str(excinfo.value) == unstructured_messages_error
    assert isinstance(guard.history.last.exception, ValidationError)
    assert guard.history.last.exception == excinfo.value


@pytest.mark.parametrize(
    "on_fail,structured_messages_error,unstructured_messages_error,",
    [
        (
            OnFailAction.REASK,
            "Messages validation failed: incorrect_value='What kind of pet should I get?' fail_results=[FailResult(outcome='fail', error_message='must be exactly two words', fix_value='What kind', error_spans=None, metadata=None, validated_chunk=None)] additional_properties={} path=None",  # noqa
            "Messages validation failed: incorrect_value='What kind of pet should I get?' fail_results=[FailResult(outcome='fail', error_message='must be exactly two words', fix_value='What kind', error_spans=None, metadata=None, validated_chunk=None)] additional_properties={} path=None",  # noqa
        ),
        (
            OnFailAction.FILTER,
            "Messages validation failed",
            "Messages validation failed",
        ),
        (
            OnFailAction.REFRAIN,
            "Messages validation failed",
            "Messages validation failed",
        ),
        (
            OnFailAction.EXCEPTION,
            "Validation failed for field with errors: must be exactly two words",
            "Validation failed for field with errors: must be exactly two words",
        ),
    ],
)
@pytest.mark.asyncio
async def test_input_validation_fail_async(
    mocker,
    on_fail,
    structured_messages_error,
    unstructured_messages_error,
):
    async def custom_llm(messages, *args, **kwargs) -> str:
        raise Exception(
            "LLM was called when it should not have been!"
            "Input Validation did not raise as expected!"
        )

    # with_messages_validation
    guard = AsyncGuard.for_pydantic(output_class=Pet)
    guard.use(TwoWords(on_fail=on_fail), on="messages")

    with pytest.raises(ValidationError) as excinfo:
        await guard(
            custom_llm,
            messages=[
                {
                    "role": "user",
                    "content": "What kind of pet should I get?",
                }
            ],
        )
    assert str(excinfo.value) == structured_messages_error
    assert isinstance(guard.history.last.exception, ValidationError)
    assert guard.history.last.exception == excinfo.value

    # with_messages_validation
    guard = AsyncGuard.for_pydantic(output_class=Pet)
    guard.use(TwoWords(on_fail=on_fail), on="messages")

    with pytest.raises(ValidationError) as excinfo:
        await guard(
            custom_llm,
            messages=[
                {
                    "role": "user",
                    "content": "What kind of pet should I get?",
                }
            ],
        )
    assert str(excinfo.value) == structured_messages_error
    assert isinstance(guard.history.last.exception, ValidationError)
    assert guard.history.last.exception == excinfo.value

    # rail prompt validation
    guard = AsyncGuard.for_rail_string(
        f"""
<rail version="0.1">
<messages
    validators="two-words"
    on-fail-two-words="{on_fail.value}"
>
<message role="user">What kind of pet should I get?</message>
</messages>
<output type="string">
</output>
</rail>
"""
    )
    with pytest.raises(ValidationError) as excinfo:
        await guard(
            custom_llm,
        )
    assert str(excinfo.value) == unstructured_messages_error
    assert isinstance(guard.history.last.exception, ValidationError)
    assert guard.history.last.exception == excinfo.value
