import json
from typing import Any, Dict, List

import pytest
from pydantic import BaseModel, Field

from guardrails import Guard, Validator, register_validator
from guardrails.async_guard import AsyncGuard
from guardrails.errors import ValidationError
from guardrails.utils.openai_utils import (
    get_static_openai_create_func,
)
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

    guard = Guard.from_pydantic(MyModel)
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

    guard = Guard.from_pydantic(MyModel)
    output = guard.parse(
        '{"a_field": "hello there yo"}',
        num_reasks=0,
    )

    assert output.validated_output == {"a_field": "hullo"}

    # (Validator, on_fail) tuple fix

    class MyModel(BaseModel):
        a_field: str = Field(..., validators=[(TwoWords(), OnFailAction.FIX)])

    guard = Guard.from_pydantic(MyModel)
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

    guard = Guard.from_pydantic(MyModel)

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

    guard = Guard.from_pydantic(MyModel)

    output = guard.parse(
        '{"a_field": "hello there yo"}',
        num_reasks=0,
    )

    assert output.validated_output == {"a_field": "hello there"}
    assert guard.history.first.iterations.first.reasks[0] == hello_reask

    # (Validator, on_fail) tuple reask

    class MyModel(BaseModel):
        a_field: str = Field(..., validators=[(TwoWords(), OnFailAction.REASK)])

    guard = Guard.from_pydantic(MyModel)

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
    guard = Guard.from_pydantic(MyModel)
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

    guard = Guard.from_rail_string(rail_str)

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


def custom_fix_on_fail_handler(value: Any, fail_results: List[FailResult]):
    return value + " " + value


def custom_reask_on_fail_handler(value: Any, fail_results: List[FailResult]):
    return FieldReAsk(incorrect_value=value, fail_results=fail_results)


def custom_exception_on_fail_handler(value: Any, fail_results: List[FailResult]):
    raise ValidationError("Something went wrong!")


def custom_filter_on_fail_handler(value: Any, fail_results: List[FailResult]):
    return Filter()


def custom_refrain_on_fail_handler(value: Any, fail_results: List[FailResult]):
    return Refrain()


@pytest.mark.parametrize(
    "custom_reask_func, expected_result",
    [
        (
            custom_fix_on_fail_handler,
            {"pet_type": "dog dog", "name": "Fido"},
        ),
        (
            custom_reask_on_fail_handler,
            FieldReAsk(
                incorrect_value="dog",
                path=["pet_type"],
                fail_results=[
                    FailResult(
                        error_message="must be exactly two words",
                        fix_value="dog dog",
                    )
                ],
            ),
        ),
        (
            custom_exception_on_fail_handler,
            ValidationError,
        ),
        (
            custom_filter_on_fail_handler,
            None,
        ),
        (
            custom_refrain_on_fail_handler,
            None,
        ),
    ],
)
# @pytest.mark.parametrize(
#     "validator_spec",
#     [
#         lambda val_func: TwoWords(on_fail=val_func),
#         # This was never supported even pre-0.5.x.
#         # Trying this with function calling will throw.
#         lambda val_func: ("two-words", val_func),
#     ],
# )
def test_custom_on_fail_handler(
    custom_reask_func,
    expected_result,
):
    prompt = """
        What kind of pet should I get and what should I name it?

        ${gr.complete_json_suffix_v2}
    """

    output = """
    {
       "pet_type": "dog",
       "name": "Fido"
    }
    """

    validator: Validator = TwoWords(on_fail=custom_reask_func)

    class Pet(BaseModel):
        pet_type: str = Field(description="Species of pet", validators=[validator])
        name: str = Field(description="a unique pet name")

    guard = Guard.from_pydantic(output_class=Pet, prompt=prompt)
    if isinstance(expected_result, type) and issubclass(expected_result, Exception):
        with pytest.raises(ValidationError) as excinfo:
            guard.parse(output, num_reasks=0)
        assert str(excinfo.value) == "Something went wrong!"
    else:
        response = guard.parse(output, num_reasks=0)
        if isinstance(expected_result, FieldReAsk):
            assert guard.history.first.iterations.first.reasks[0] == expected_result
        else:
            assert response.validated_output == expected_result


class TestCustomOnFailHandler:
    def test_custom_fix(self):
        prompt = """
            What kind of pet should I get and what should I name it?

            ${gr.complete_json_suffix_v2}
        """

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

        guard = Guard.from_pydantic(output_class=Pet, prompt=prompt)

        response = guard.parse(output, num_reasks=0)
        assert response.validation_passed is True
        assert response.validated_output == expected_result

    def test_custom_reask(self):
        prompt = """
            What kind of pet should I get and what should I name it?

            ${gr.complete_json_suffix_v2}
        """

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

        guard = Guard.from_pydantic(output_class=Pet, prompt=prompt)

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

        guard = Guard.from_pydantic(output_class=Pet, prompt=prompt)

        with pytest.raises(ValidationError) as excinfo:
            guard.parse(output, num_reasks=0)
        assert str(excinfo.value) == "Something went wrong!"

    def test_custom_filter(self):
        prompt = """
            What kind of pet should I get and what should I name it?

            ${gr.complete_json_suffix_v2}
        """

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

        guard = Guard.from_pydantic(output_class=Pet, prompt=prompt)

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

        guard = Guard.from_pydantic(output_class=Pet, prompt=prompt)

        response = guard.parse(output, num_reasks=0)

        assert response.validation_passed is False
        assert response.validated_output is None


class Pet(BaseModel):
    name: str = Field(description="a unique pet name")


def test_input_validation_fix(mocker):
    def mock_llm_api(*args, **kwargs):
        return json.dumps({"name": "Fluffy"})

    # fix returns an amended value for prompt/instructions validation,
    guard = Guard.from_pydantic(output_class=Pet)
    guard.use(TwoWords(on_fail=OnFailAction.FIX), on="prompt")

    guard(
        mock_llm_api,
        prompt="What kind of pet should I get?",
    )
    assert (
        guard.history.first.iterations.first.outputs.validation_response == "What kind"
    )
    guard = Guard.from_pydantic(output_class=Pet)
    guard.use(TwoWords(on_fail=OnFailAction.FIX), on="instructions")

    guard(
        mock_llm_api,
        prompt="What kind of pet should I get and what should I name it?",
        instructions="But really, what kind of pet should I get?",
    )
    assert (
        guard.history.first.iterations.first.outputs.validation_response
        == "But really,"
    )

    # but raises for msg_history validation
    guard = Guard.from_pydantic(output_class=Pet)
    guard.use(TwoWords(on_fail=OnFailAction.FIX), on="msg_history")

    with pytest.raises(ValidationError) as excinfo:
        guard(
            mock_llm_api,
            msg_history=[
                {
                    "role": "user",
                    "content": "What kind of pet should I get?",
                }
            ],
        )
    assert str(excinfo.value) == "Message history validation failed"
    assert isinstance(guard.history.first.exception, ValidationError)
    assert guard.history.first.exception == excinfo.value

    # rail prompt validation
    guard = Guard.from_rail_string(
        """
<rail version="0.1">
<prompt
    validators="two-words"
    on-fail-two-words="fix"
>
This is not two words
</prompt>
<output type="string">
</output>
</rail>
"""
    )
    guard(
        mock_llm_api,
    )
    assert guard.history.first.iterations.first.outputs.validation_response == "This is"

    # rail instructions validation
    guard = Guard.from_rail_string(
        """
<rail version="0.1">
<prompt>
This is not two words
</prompt>
<instructions
    validators="two-words"
    on-fail-two-words="fix"
>
This also is not two words
</instructions>
<output type="string">
</output>
</rail>
"""
    )
    guard(
        mock_llm_api,
    )
    assert (
        guard.history.first.iterations.first.outputs.validation_response == "This also"
    )


@pytest.mark.asyncio
async def test_async_input_validation_fix(mocker):
    async def mock_llm_api(*args, **kwargs):
        return json.dumps({"name": "Fluffy"})

    # fix returns an amended value for prompt/instructions validation,
    guard = AsyncGuard.from_pydantic(output_class=Pet)
    guard.use(TwoWords(on_fail=OnFailAction.FIX), on="prompt")

    await guard(
        mock_llm_api,
        prompt="What kind of pet should I get?",
    )
    assert (
        guard.history.first.iterations.first.outputs.validation_response == "What kind"
    )

    guard = AsyncGuard.from_pydantic(output_class=Pet)
    guard.use(TwoWords(on_fail=OnFailAction.FIX), on="instructions")

    await guard(
        mock_llm_api,
        prompt="What kind of pet should I get and what should I name it?",
        instructions="But really, what kind of pet should I get?",
    )
    assert (
        guard.history.first.iterations.first.outputs.validation_response
        == "But really,"
    )

    # but raises for msg_history validation
    guard = AsyncGuard.from_pydantic(output_class=Pet)
    guard.use(TwoWords(on_fail=OnFailAction.FIX), on="msg_history")

    with pytest.raises(ValidationError) as excinfo:
        await guard(
            mock_llm_api,
            msg_history=[
                {
                    "role": "user",
                    "content": "What kind of pet should I get?",
                }
            ],
        )
    assert str(excinfo.value) == "Message history validation failed"
    assert isinstance(guard.history.first.exception, ValidationError)
    assert guard.history.first.exception == excinfo.value

    # rail prompt validation
    guard = AsyncGuard.from_rail_string(
        """
<rail version="0.1">
<prompt
    validators="two-words"
    on-fail-two-words="fix"
>
This is not two words
</prompt>
<output type="string">
</output>
</rail>
"""
    )
    await guard(
        mock_llm_api,
    )
    assert guard.history.first.iterations.first.outputs.validation_response == "This is"

    # rail instructions validation
    guard = AsyncGuard.from_rail_string(
        """
<rail version="0.1">
<prompt>
This is not two words
</prompt>
<instructions
    validators="two-words"
    on-fail-two-words="fix"
>
This also is not two words
</instructions>
<output type="string">
</output>
</rail>
"""
    )
    await guard(
        mock_llm_api,
    )
    assert (
        guard.history.first.iterations.first.outputs.validation_response == "This also"
    )


@pytest.mark.parametrize(
    "on_fail,"
    "structured_prompt_error,"
    "structured_instructions_error,"
    "structured_message_history_error,"
    "unstructured_prompt_error,"
    "unstructured_instructions_error",
    [
        (
            OnFailAction.REASK,
            "Prompt validation failed: incorrect_value='What kind of pet should I get?' fail_results=[FailResult(outcome='fail', error_message='must be exactly two words', fix_value='What kind', error_spans=None, metadata=None, validated_chunk=None)] additional_properties={} path=None",  # noqa
            "Instructions validation failed: incorrect_value='What kind of pet should I get?' fail_results=[FailResult(outcome='fail', error_message='must be exactly two words', fix_value='What kind', error_spans=None, metadata=None, validated_chunk=None)] additional_properties={} path=None",  # noqa
            "Message history validation failed: incorrect_value='What kind of pet should I get?' fail_results=[FailResult(outcome='fail', error_message='must be exactly two words', fix_value='What kind', error_spans=None, metadata=None, validated_chunk=None)] additional_properties={} path=None",  # noqa
            "Prompt validation failed: incorrect_value='\\nThis is not two words\\n' fail_results=[FailResult(outcome='fail', error_message='must be exactly two words', fix_value='This is', error_spans=None, metadata=None, validated_chunk=None)] additional_properties={} path=None",  # noqa
            "Instructions validation failed: incorrect_value='\\nThis also is not two words\\n' fail_results=[FailResult(outcome='fail', error_message='must be exactly two words', fix_value='This also', error_spans=None, metadata=None, validated_chunk=None)] additional_properties={} path=None",  # noqa
        ),
        (
            OnFailAction.FILTER,
            "Prompt validation failed",
            "Instructions validation failed",
            "Message history validation failed",
            "Prompt validation failed",
            "Instructions validation failed",
        ),
        (
            OnFailAction.REFRAIN,
            "Prompt validation failed",
            "Instructions validation failed",
            "Message history validation failed",
            "Prompt validation failed",
            "Instructions validation failed",
        ),
        (
            OnFailAction.EXCEPTION,
            "Validation failed for field with errors: must be exactly two words",
            "Validation failed for field with errors: must be exactly two words",
            "Validation failed for field with errors: must be exactly two words",
            "Validation failed for field with errors: must be exactly two words",
            "Validation failed for field with errors: must be exactly two words",
        ),
    ],
)
def test_input_validation_fail(
    on_fail,
    structured_prompt_error,
    structured_instructions_error,
    structured_message_history_error,
    unstructured_prompt_error,
    unstructured_instructions_error,
):
    # With Prompt Validation
    guard = Guard.from_pydantic(output_class=Pet)
    guard.use(TwoWords(on_fail=on_fail), on="prompt")

    def custom_llm(*args, **kwargs):
        raise Exception(
            "LLM was called when it should not have been!"
            "Input Validation did not raise as expected!"
        )

    with pytest.raises(ValidationError) as excinfo:
        guard(
            custom_llm,
            prompt="What kind of pet should I get?",
        )
    assert str(excinfo.value) == structured_prompt_error
    assert isinstance(guard.history.last.exception, ValidationError)
    assert guard.history.last.exception == excinfo.value

    # With Instructions Validation
    guard = Guard.from_pydantic(output_class=Pet)
    guard.use(TwoWords(on_fail=on_fail), on="instructions")

    with pytest.raises(ValidationError) as excinfo:
        guard(
            custom_llm,
            prompt="What kind of pet should I get and what should I name it?",
            instructions="What kind of pet should I get?",
        )

    assert str(excinfo.value) == structured_instructions_error
    assert isinstance(guard.history.last.exception, ValidationError)
    assert guard.history.last.exception == excinfo.value

    # With Msg History Validation
    guard = Guard.from_pydantic(output_class=Pet)
    guard.use(TwoWords(on_fail=on_fail), on="msg_history")

    with pytest.raises(ValidationError) as excinfo:
        guard(
            custom_llm,
            msg_history=[
                {
                    "role": "user",
                    "content": "What kind of pet should I get?",
                }
            ],
        )
    assert str(excinfo.value) == structured_message_history_error
    assert isinstance(guard.history.last.exception, ValidationError)
    assert guard.history.last.exception == excinfo.value

    # Rail Prompt Validation
    guard = Guard.from_rail_string(
        f"""
<rail version="0.1">
<prompt
    validators="two-words"
    on-fail-two-words="{on_fail.value}"
>
This is not two words
</prompt>
<output type="string">
</output>
</rail>
"""
    )
    with pytest.raises(ValidationError) as excinfo:
        guard(
            custom_llm,
        )
    assert str(excinfo.value) == unstructured_prompt_error
    assert isinstance(guard.history.last.exception, ValidationError)
    assert guard.history.last.exception == excinfo.value

    # Rail Instructions Validation
    guard = Guard.from_rail_string(
        f"""
<rail version="0.1">
<prompt>
This is not two words
</prompt>
<instructions
    validators="two-words"
    on-fail-two-words="{on_fail.value}"
>
This also is not two words
</instructions>
<output type="string">
</output>
</rail>
"""
    )
    with pytest.raises(ValidationError) as excinfo:
        guard(
            custom_llm,
        )
    assert str(excinfo.value) == unstructured_instructions_error
    assert isinstance(guard.history.last.exception, ValidationError)
    assert guard.history.last.exception == excinfo.value


@pytest.mark.parametrize(
    "on_fail,"
    "structured_prompt_error,"
    "structured_instructions_error,"
    "structured_message_history_error,"
    "unstructured_prompt_error,"
    "unstructured_instructions_error",
    [
        (
            OnFailAction.REASK,
            "Prompt validation failed: incorrect_value='What kind of pet should I get?\\n\\nJson Output:\\n\\n' fail_results=[FailResult(outcome='fail', error_message='must be exactly two words', fix_value='What kind', error_spans=None, metadata=None, validated_chunk=None)] additional_properties={} path=None",  # noqa
            "Instructions validation failed: incorrect_value='What kind of pet should I get?' fail_results=[FailResult(outcome='fail', error_message='must be exactly two words', fix_value='What kind', error_spans=None, metadata=None, validated_chunk=None)] additional_properties={} path=None",  # noqa
            "Message history validation failed: incorrect_value='What kind of pet should I get?' fail_results=[FailResult(outcome='fail', error_message='must be exactly two words', fix_value='What kind', error_spans=None, metadata=None, validated_chunk=None)] additional_properties={} path=None",  # noqa
            "Prompt validation failed: incorrect_value='\\nThis is not two words\\n\\n\\nString Output:\\n\\n' fail_results=[FailResult(outcome='fail', error_message='must be exactly two words', fix_value='This is', error_spans=None, metadata=None, validated_chunk=None)] additional_properties={} path=None",  # noqa
            "Instructions validation failed: incorrect_value='\\nThis also is not two words\\n' fail_results=[FailResult(outcome='fail', error_message='must be exactly two words', fix_value='This also', error_spans=None, metadata=None, validated_chunk=None)] additional_properties={} path=None",  # noqa
        ),
        (
            OnFailAction.FILTER,
            "Prompt validation failed",
            "Instructions validation failed",
            "Message history validation failed",
            "Prompt validation failed",
            "Instructions validation failed",
        ),
        (
            OnFailAction.REFRAIN,
            "Prompt validation failed",
            "Instructions validation failed",
            "Message history validation failed",
            "Prompt validation failed",
            "Instructions validation failed",
        ),
        (
            OnFailAction.EXCEPTION,
            "Validation failed for field with errors: must be exactly two words",
            "Validation failed for field with errors: must be exactly two words",
            "Validation failed for field with errors: must be exactly two words",
            "Validation failed for field with errors: must be exactly two words",
            "Validation failed for field with errors: must be exactly two words",
        ),
    ],
)
@pytest.mark.asyncio
async def test_input_validation_fail_async(
    mocker,
    on_fail,
    structured_prompt_error,
    structured_instructions_error,
    structured_message_history_error,
    unstructured_prompt_error,
    unstructured_instructions_error,
):
    async def custom_llm(*args, **kwargs):
        raise Exception(
            "LLM was called when it should not have been!"
            "Input Validation did not raise as expected!"
        )

    mocker.patch(
        "guardrails.llm_providers.get_static_openai_acreate_func",
        return_value=custom_llm,
    )

    # with_prompt_validation
    guard = AsyncGuard.from_pydantic(output_class=Pet)
    guard.use(TwoWords(on_fail=on_fail), on="prompt")

    with pytest.raises(ValidationError) as excinfo:
        await guard(
            custom_llm,
            prompt="What kind of pet should I get?",
        )
    assert str(excinfo.value) == structured_prompt_error
    assert isinstance(guard.history.last.exception, ValidationError)
    assert guard.history.last.exception == excinfo.value

    # with_instructions_validation
    guard = AsyncGuard.from_pydantic(output_class=Pet)
    guard.use(TwoWords(on_fail=on_fail), on="instructions")

    with pytest.raises(ValidationError) as excinfo:
        await guard(
            custom_llm,
            prompt="What kind of pet should I get and what should I name it?",
            instructions="What kind of pet should I get?",
        )
    assert str(excinfo.value) == structured_instructions_error
    assert isinstance(guard.history.last.exception, ValidationError)
    assert guard.history.last.exception == excinfo.value

    # with_msg_history_validation
    guard = AsyncGuard.from_pydantic(output_class=Pet)
    guard.use(TwoWords(on_fail=on_fail), on="msg_history")

    with pytest.raises(ValidationError) as excinfo:
        await guard(
            custom_llm,
            msg_history=[
                {
                    "role": "user",
                    "content": "What kind of pet should I get?",
                }
            ],
        )
    assert str(excinfo.value) == structured_message_history_error
    assert isinstance(guard.history.last.exception, ValidationError)
    assert guard.history.last.exception == excinfo.value

    # with_messages_validation
    guard = AsyncGuard.from_pydantic(output_class=Pet)
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
    assert str(excinfo.value) == structured_message_history_error
    assert isinstance(guard.history.last.exception, ValidationError)
    assert guard.history.last.exception == excinfo.value

    # rail prompt validation
    guard = AsyncGuard.from_rail_string(
        f"""
<rail version="0.1">
<prompt
    validators="two-words"
    on-fail-two-words="{on_fail.value}"
>
This is not two words
</prompt>
<output type="string">
</output>
</rail>
"""
    )
    with pytest.raises(ValidationError) as excinfo:
        await guard(
            custom_llm,
        )
    assert str(excinfo.value) == unstructured_prompt_error
    assert isinstance(guard.history.last.exception, ValidationError)
    assert guard.history.last.exception == excinfo.value

    # rail instructions validation
    guard = AsyncGuard.from_rail_string(
        f"""
<rail version="0.1">
<prompt>
This is not two words
</prompt>
<instructions
    validators="two-words"
    on-fail-two-words="{on_fail.value}"
>
This also is not two words
</instructions>
<output type="string">
</output>
</rail>
"""
    )
    with pytest.raises(ValidationError) as excinfo:
        await guard(
            custom_llm,
        )
    assert str(excinfo.value) == unstructured_instructions_error
    assert isinstance(guard.history.last.exception, ValidationError)
    assert guard.history.last.exception == excinfo.value


def test_input_validation_mismatch_raise():
    # prompt validation, msg_history argument
    guard = Guard.from_pydantic(output_class=Pet)
    guard.use(TwoWords(on_fail=OnFailAction.FIX), on="prompt")

    with pytest.raises(ValueError):
        guard(
            get_static_openai_create_func(),
            msg_history=[
                {
                    "role": "user",
                    "content": "What kind of pet should I get?",
                }
            ],
        )

    # instructions validation, msg_history argument
    guard = Guard.from_pydantic(output_class=Pet)
    guard.use(TwoWords(on_fail=OnFailAction.FIX), on="instructions")

    with pytest.raises(ValueError):
        guard(
            get_static_openai_create_func(),
            msg_history=[
                {
                    "role": "user",
                    "content": "What kind of pet should I get?",
                }
            ],
        )

    # msg_history validation, prompt argument
    guard = Guard.from_pydantic(output_class=Pet)
    guard.use(TwoWords(on_fail=OnFailAction.FIX), on="msg_history")

    with pytest.raises(ValueError):
        guard(
            get_static_openai_create_func(),
            prompt="What kind of pet should I get?",
        )
