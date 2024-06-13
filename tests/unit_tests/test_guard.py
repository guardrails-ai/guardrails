import importlib
import pytest

import openai
from pydantic import BaseModel
from guardrails import Guard, Validator
from guardrails.utils.validator_utils import verify_metadata_requirements
from guardrails.utils import args, kwargs, on_fail
from guardrails.utils.openai_utils import OPENAI_VERSION
from guardrails.validator_base import OnFailAction
from guardrails.validators import (  # ReadingTime,
    EndsWith,
    LowerCase,
    OneLine,
    PassResult,
    TwoWords,
    UpperCase,
    ValidLength,
    register_validator,
)


@register_validator("myrequiringvalidator", data_type="string")
class RequiringValidator(Validator):
    required_metadata_keys = ["required_key"]

    def validate(self, value, metadata):
        return PassResult()


@register_validator("myrequiringvalidator2", data_type="string")
class RequiringValidator2(Validator):
    required_metadata_keys = ["required_key2"]

    def validate(self, value, metadata):
        return PassResult()


@pytest.mark.parametrize(
    "spec,metadata,error_message",
    [
        (
            """
<rail version="0.1">
<output>
    <string name="string_name" validators="myrequiringvalidator" />
</output>
</rail>
        """,
            {"required_key": "a"},
            "Missing required metadata keys: required_key",
        ),
        (
            """
<rail version="0.1">
<output>
    <object name="temp_name">
        <string name="string_name" validators="myrequiringvalidator" />
    </object>
    <list name="list_name">
        <string name="string_name" validators="myrequiringvalidator2" />
    </list>
</output>
</rail>
        """,
            {"required_key": "a", "required_key2": "b"},
            "Missing required metadata keys: required_key, required_key2",
        ),
        (
            """
<rail version="0.1">
<output>
    <object name="temp_name">
    <list name="list_name">
    <choice name="choice_name" discriminator="hi">
    <case name="hello">
        <string name="string_name" />
    </case>
    <case name="hiya">
        <string name="string_name" validators="myrequiringvalidator" />
    </case>
    </choice>
    </list>
    </object>
</output>
</rail>
""",
            {"required_key": "a"},
            "Missing required metadata keys: required_key",
        ),
    ],
)
@pytest.mark.asyncio
@pytest.mark.skipif(not OPENAI_VERSION.startswith("0"), reason="Only for OpenAI v0")
async def test_required_metadata(spec, metadata, error_message):
    guard = Guard.from_rail_string(spec)

    missing_keys = verify_metadata_requirements({}, guard.output_schema.root_datatype)
    assert set(missing_keys) == set(metadata)

    not_missing_keys = verify_metadata_requirements(
        metadata, guard.output_schema.root_datatype
    )
    assert not_missing_keys == []

    # test sync guard
    with pytest.raises(ValueError) as excinfo:
        guard.parse("{}")
    assert str(excinfo.value) == error_message

    response = guard.parse("{}", metadata=metadata, num_reasks=0)
    assert response.error is None

    # test async guard
    with pytest.raises(ValueError) as excinfo:
        guard.parse("{}")
        await guard.parse("{}", llm_api=openai.ChatCompletion.acreate, num_reasks=0)
    assert str(excinfo.value) == error_message

    response = await guard.parse(
        "{}", metadata=metadata, llm_api=openai.ChatCompletion.acreate, num_reasks=0
    )
    assert response.error is None


empty_rail_string = """<rail version="0.1">
<output
    type="string"
    description="empty railspec"
/>
</rail>"""


class EmptyModel(BaseModel):
    empty_field: str


# FIXME: Init with json schema
# i_guard_none = Guard(rail)
# i_guard_two = Guard(rail, 2)
r_guard_none = Guard.from_rail("tests/unit_tests/test_assets/empty.rail")
r_guard_two = Guard.from_rail("tests/unit_tests/test_assets/empty.rail", num_reasks=2)
rs_guard_none = Guard.from_rail_string(empty_rail_string)
rs_guard_two = Guard.from_rail_string(empty_rail_string, num_reasks=2)
py_guard_none = Guard.from_pydantic(output_class=EmptyModel)
py_guard_two = Guard.from_pydantic(output_class=EmptyModel, num_reasks=2)
s_guard_none = Guard.from_string(validators=[], string_description="empty railspec")
s_guard_two = Guard.from_string(
    validators=[], description="empty railspec", num_reasks=2
)


class TestConfigure:
    def test_num_reasks(self):
        guard = Guard()
        guard.configure()

        assert guard._num_reasks is None

        guard.configure(num_reasks=2)

        assert guard._num_reasks == 2

    def test_tracer(self, mocker):
        mocker.patch("guardrails.guard.Tracer")
        mock_set_tracer = mocker.patch("guardrails.guard.set_tracer")
        mock_set_tracer_context = mocker.patch("guardrails.guard.set_tracer_context")
        mock_get_tracer_context = mocker.patch("guardrails.guard.get_tracer_context")

        from guardrails.guard import Tracer, Guard

        guard = Guard()
        tracer = Tracer()

        guard.configure()

        assert guard._tracer is None

        guard.configure(tracer=tracer)

        assert guard._tracer == tracer

        assert mock_set_tracer.call_count == 1
        assert mock_set_tracer_context.call_count == 1
        assert mock_get_tracer_context.call_count == 1


def guard_init_from_rail():
    guard = Guard.from_rail("tests/unit_tests/test_assets/simple.rail")
    assert (
        guard.instructions.format().source.strip()
        == "You are a helpful bot, who answers only with valid JSON"
    )
    assert guard.prompt.format().source.strip() == "Extract a string from the text"


def test_use():
    guard: Guard = (
        Guard()
        .use(EndsWith("a"))
        .use(OneLine())
        .use(LowerCase)
        .use(TwoWords, on_fail=OnFailAction.REASK)
        .use(ValidLength, 0, 12, on_fail=OnFailAction.REFRAIN)
    )

    # print(guard.__stringify__())
    assert len(guard._validators) == 5

    assert isinstance(guard._validators[0], EndsWith)
    assert guard._validators[0]._kwargs["end"] == "a"
    assert (
        guard._validators[0].on_fail_descriptor == OnFailAction.FIX
    )  # bc this is the default

    assert isinstance(guard._validators[1], OneLine)
    assert (
        guard._validators[1].on_fail_descriptor == OnFailAction.NOOP
    )  # bc this is the default

    assert isinstance(guard._validators[2], LowerCase)
    assert (
        guard._validators[2].on_fail_descriptor == OnFailAction.NOOP
    )  # bc this is the default

    assert isinstance(guard._validators[3], TwoWords)
    assert guard._validators[3].on_fail_descriptor == OnFailAction.REASK  # bc we set it

    assert isinstance(guard._validators[4], ValidLength)
    assert guard._validators[4]._min == 0
    assert guard._validators[4]._kwargs["min"] == 0
    assert guard._validators[4]._max == 12
    assert guard._validators[4]._kwargs["max"] == 12
    assert (
        guard._validators[4].on_fail_descriptor == OnFailAction.REFRAIN
    )  # bc we set it

    # No longer a constraint
    # # Raises error when trying to `use` a validator on a non-string
    # with pytest.raises(RuntimeError):

    class TestClass(BaseModel):
        another_field: str

    py_guard = Guard.from_pydantic(output_class=TestClass)
    py_guard.use(EndsWith("a"))
    assert py_guard._validator_map.get("$") == [EndsWith("a")]

    # Use a combination of prompt, instructions, msg_history and output validators
    # Should only have the output validators in the guard,
    # everything else is in the schema
    guard: Guard = (
        Guard()
        .use(LowerCase, on="prompt")
        .use(OneLine, on="prompt")
        .use(UpperCase, on="instructions")
        .use(LowerCase, on="msg_history")
        .use(
            EndsWith, end="a", on="output"
        )  # default on="output", still explicitly set
        .use(
            TwoWords, on_fail=OnFailAction.REASK
        )  # default on="output", implicitly set
    )

    # Check schemas for prompt, instructions and msg_history validators
    prompt_validators = guard._validator_map.get("prompt", [])
    assert len(prompt_validators) == 2
    assert prompt_validators[0].__class__.__name__ == "LowerCase"
    assert prompt_validators[1].__class__.__name__ == "OneLine"

    instructions_validators = guard._validator_map.get("instructions", [])
    assert len(instructions_validators) == 1
    assert instructions_validators[0].__class__.__name__ == "UpperCase"

    msg_history_validators = guard._validator_map.get("msg_history", [])
    assert len(msg_history_validators) == 1
    assert msg_history_validators[0].__class__.__name__ == "LowerCase"

    # Check guard for output validators
    assert len(guard._validators) == 6

    assert isinstance(guard._validators[4], EndsWith)
    assert guard._validators[4]._kwargs["end"] == "a"
    assert (
        guard._validators[4].on_fail_descriptor == OnFailAction.FIX
    )  # bc this is the default

    assert isinstance(guard._validators[5], TwoWords)
    assert guard._validators[5].on_fail_descriptor == OnFailAction.REASK  # bc we set it

    # Test with an unrecognized "on" parameter, should warn with a UserWarning
    with pytest.warns(UserWarning):
        guard: Guard = (
            Guard()
            .use(EndsWith("a"), on="response")  # invalid on parameter
            .use(OneLine, on="prompt")  # valid on parameter
        )


def test_use_many_instances():
    guard: Guard = Guard().use_many(
        EndsWith("a"), OneLine(), LowerCase(), TwoWords(on_fail=OnFailAction.REASK)
    )

    # print(guard.__stringify__())
    assert len(guard._validators) == 4

    assert isinstance(guard._validators[0], EndsWith)
    assert guard._validators[0]._end == "a"
    assert guard._validators[0]._kwargs["end"] == "a"
    assert (
        guard._validators[0].on_fail_descriptor == OnFailAction.FIX
    )  # bc this is the default

    assert isinstance(guard._validators[1], OneLine)
    assert (
        guard._validators[1].on_fail_descriptor == OnFailAction.NOOP
    )  # bc this is the default

    assert isinstance(guard._validators[2], LowerCase)
    assert (
        guard._validators[2].on_fail_descriptor == OnFailAction.NOOP
    )  # bc this is the default

    assert isinstance(guard._validators[3], TwoWords)
    assert guard._validators[3].on_fail_descriptor == OnFailAction.REASK  # bc we set it

    # No longer a constraint
    # # Raises error when trying to `use_many` a validator on a non-string
    # with pytest.raises(RuntimeError):

    class TestClass(BaseModel):
        another_field: str

    py_guard = Guard.from_pydantic(output_class=TestClass)
    py_guard.use_many(
        EndsWith("a"),
        OneLine(),
        LowerCase(),
        TwoWords(on_fail=OnFailAction.REASK),
    )

    assert py_guard._validator_map.get("$") == [
        EndsWith("a"),
        OneLine(),
        LowerCase(),
        TwoWords(on_fail=OnFailAction.REASK),
    ]

    # Test with explicitly setting the "on" parameter = "output"
    guard: Guard = Guard().use_many(
        EndsWith("a"),
        OneLine(),
        LowerCase(),
        TwoWords(on_fail=OnFailAction.REASK),
        on="output",
    )

    assert len(guard._validators) == 4  # still 4 output validators, hence 4

    assert isinstance(guard._validators[0], EndsWith)
    assert guard._validators[0]._end == "a"
    assert guard._validators[0]._kwargs["end"] == "a"
    assert (
        guard._validators[0].on_fail_descriptor == OnFailAction.FIX
    )  # bc this is the default

    assert isinstance(guard._validators[1], OneLine)
    assert (
        guard._validators[1].on_fail_descriptor == OnFailAction.NOOP
    )  # bc this is the default

    assert isinstance(guard._validators[2], LowerCase)
    assert (
        guard._validators[2].on_fail_descriptor == OnFailAction.NOOP
    )  # bc this is the default

    assert isinstance(guard._validators[3], TwoWords)
    assert guard._validators[3].on_fail_descriptor == OnFailAction.REASK  # bc we set it

    # Test with explicitly setting the "on" parameter = "prompt"
    guard: Guard = Guard().use_many(
        OneLine(), LowerCase(), TwoWords(on_fail=OnFailAction.REASK), on="prompt"
    )

    prompt_validators = guard._validator_map.get("prompt", [])
    assert len(prompt_validators) == 3
    assert prompt_validators[0].__class__.__name__ == "OneLine"
    assert prompt_validators[1].__class__.__name__ == "LowerCase"
    assert prompt_validators[2].__class__.__name__ == "TwoWords"
    assert len(guard._validators) == 3

    # Test with explicitly setting the "on" parameter = "instructions"
    guard: Guard = Guard().use_many(
        OneLine(), LowerCase(), TwoWords(on_fail=OnFailAction.REASK), on="instructions"
    )

    instructions_validators = guard._validator_map.get("instructions", [])
    assert len(instructions_validators) == 3
    assert instructions_validators[0].__class__.__name__ == "OneLine"
    assert instructions_validators[1].__class__.__name__ == "LowerCase"
    assert instructions_validators[2].__class__.__name__ == "TwoWords"
    assert len(guard._validators) == 3

    # Test with explicitly setting the "on" parameter = "msg_history"
    guard: Guard = Guard().use_many(
        OneLine(), LowerCase(), TwoWords(on_fail=OnFailAction.REASK), on="msg_history"
    )

    msg_history_validators = guard._validator_map.get("msg_history", [])
    assert len(msg_history_validators) == 3
    assert msg_history_validators[0].__class__.__name__ == "OneLine"
    assert msg_history_validators[1].__class__.__name__ == "LowerCase"
    assert msg_history_validators[2].__class__.__name__ == "TwoWords"
    assert len(guard._validators) == 3

    # Test with an unrecognized "on" parameter, should warn with a UserWarning
    with pytest.warns(UserWarning):
        guard: Guard = Guard().use_many(
            EndsWith("a", on_fail=OnFailAction.EXCEPTION), OneLine(), on="response"
        )


def test_use_many_tuple():
    guard: Guard = Guard().use_many(
        OneLine,
        (EndsWith, ["a"], {"on_fail": OnFailAction.EXCEPTION}),
        (LowerCase, kwargs(on_fail=OnFailAction.FIX_REASK, some_other_kwarg="kwarg")),
        (TwoWords, on_fail(OnFailAction.REASK)),
        (ValidLength, args(0, 12), kwargs(on_fail=OnFailAction.REFRAIN)),
    )

    # print(guard.__stringify__())
    assert len(guard._validators) == 5

    assert isinstance(guard._validators[0], OneLine)
    assert (
        guard._validators[0].on_fail_descriptor == OnFailAction.NOOP
    )  # bc this is the default

    assert isinstance(guard._validators[1], EndsWith)
    assert guard._validators[1]._end == "a"
    assert guard._validators[1]._kwargs["end"] == "a"
    assert (
        guard._validators[1].on_fail_descriptor == OnFailAction.EXCEPTION
    )  # bc we set it

    assert isinstance(guard._validators[2], LowerCase)
    assert guard._validators[2]._kwargs["some_other_kwarg"] == "kwarg"
    assert (
        guard._validators[2].on_fail_descriptor == OnFailAction.FIX_REASK
    )  # bc this is the default

    assert isinstance(guard._validators[3], TwoWords)
    assert guard._validators[3].on_fail_descriptor == OnFailAction.REASK  # bc we set it

    assert isinstance(guard._validators[4], ValidLength)
    assert guard._validators[4]._min == 0
    assert guard._validators[4]._kwargs["min"] == 0
    assert guard._validators[4]._max == 12
    assert guard._validators[4]._kwargs["max"] == 12
    assert (
        guard._validators[4].on_fail_descriptor == OnFailAction.REFRAIN
    )  # bc we set it

    # Test with explicitly setting the "on" parameter
    guard: Guard = Guard().use_many(
        (EndsWith, ["a"], {"on_fail": OnFailAction.EXCEPTION}),
        OneLine,
        on="output",
    )

    assert len(guard._validators) == 2  # only 2 output validators, hence 2

    assert isinstance(guard._validators[0], EndsWith)
    assert guard._validators[0]._end == "a"
    assert guard._validators[0]._kwargs["end"] == "a"
    assert (
        guard._validators[0].on_fail_descriptor == OnFailAction.EXCEPTION
    )  # bc we set it

    assert isinstance(guard._validators[1], OneLine)
    assert (
        guard._validators[1].on_fail_descriptor == OnFailAction.NOOP
    )  # bc this is the default

    # Test with an unrecognized "on" parameter, should warn with a UserWarning
    with pytest.warns(UserWarning):
        guard: Guard = Guard().use_many(
            (EndsWith, ["a"], {"on_fail": OnFailAction.EXCEPTION}),
            OneLine,
            on="response",
        )


def test_validate():
    guard: Guard = (
        Guard()
        .use(OneLine)
        .use(
            LowerCase(on_fail=OnFailAction.FIX), on="output"
        )  # default on="output", still explicitly set
        .use(TwoWords)
        .use(ValidLength, 0, 12, on_fail=OnFailAction.REFRAIN)
    )

    llm_output: str = "Oh Canada"  # bc it meets our criteria

    response = guard.validate(llm_output)

    assert response.validation_passed is True
    assert response.validated_output == llm_output.lower()

    llm_output_2 = "Star Spangled Banner"  # to stick with the theme

    response_2 = guard.validate(llm_output_2)

    assert response_2.validation_passed is False
    assert response_2.validated_output is None

    # Test with a combination of prompt, output, instructions and msg_history validators
    # Should still only use the output validators to validate the output
    guard: Guard = (
        Guard()
        .use(OneLine, on="prompt")
        .use(LowerCase, on="instructions")
        .use(UpperCase, on="msg_history")
        .use(LowerCase, on="output", on_fail=OnFailAction.FIX)
        .use(TwoWords, on="output")
        .use(ValidLength, 0, 12, on="output")
    )

    llm_output: str = "Oh Canada"  # bc it meets our criteria

    response = guard.validate(llm_output)

    assert response.validation_passed is True
    assert response.validated_output == llm_output.lower()

    llm_output_2 = "Star Spangled Banner"  # to stick with the theme

    response_2 = guard.validate(llm_output_2)

    assert response_2.validation_passed is False
    assert response_2.validated_output is None


def test_use_and_use_many():
    guard: Guard = (
        Guard()
        .use_many(OneLine(), LowerCase(), on="prompt")
        .use(UpperCase, on="instructions")
        .use(LowerCase, on="msg_history")
        .use_many(
            TwoWords(on_fail=OnFailAction.REASK),
            ValidLength(0, 12, on_fail=OnFailAction.REFRAIN),
            on="output",
        )
    )

    # Check schemas for prompt, instructions and msg_history validators
    prompt_validators = guard._validator_map.get("prompt", [])
    assert len(prompt_validators) == 2
    assert prompt_validators[0].__class__.__name__ == "OneLine"
    assert prompt_validators[1].__class__.__name__ == "LowerCase"

    instructions_validators = guard._validator_map.get("instructions", [])
    assert len(instructions_validators) == 1
    assert instructions_validators[0].__class__.__name__ == "UpperCase"

    msg_history_validators = guard._validator_map.get("msg_history", [])
    assert len(msg_history_validators) == 1
    assert msg_history_validators[0].__class__.__name__ == "LowerCase"

    # Check guard for validators
    assert len(guard._validators) == 6

    assert isinstance(guard._validators[4], TwoWords)
    assert guard._validators[4].on_fail_descriptor == OnFailAction.REASK  # bc we set it

    assert isinstance(guard._validators[5], ValidLength)
    assert guard._validators[5]._min == 0
    assert guard._validators[5]._kwargs["min"] == 0
    assert guard._validators[5]._max == 12
    assert guard._validators[5]._kwargs["max"] == 12
    assert (
        guard._validators[5].on_fail_descriptor == OnFailAction.REFRAIN
    )  # bc we set it

    # Test with an unrecognized "on" parameter, should warn with a UserWarning
    with pytest.warns(UserWarning):
        guard: Guard = (
            Guard()
            .use_many(OneLine(), LowerCase(), on="prompt")
            .use(UpperCase, on="instructions")
            .use(LowerCase, on="msg_history")
            .use_many(
                TwoWords(on_fail=OnFailAction.REASK),
                ValidLength(0, 12, on_fail=OnFailAction.REFRAIN),
                on="response",  # invalid "on" parameter
            )
        )


# def test_call():
#     five_seconds = 5 / 60
#     response = Guard().use_many(
#         ReadingTime(five_seconds, on_fail=OnFailAction.EXCEPTION),
#         OneLine,
#         (EndsWith, ["a"], {"on_fail": OnFailAction.EXCEPTION}),
#         (LowerCase, kwargs(on_fail=OnFailAction.FIX_REASK, some_other_kwarg="kwarg")),
#         (TwoWords, on_fail(OnFailAction.REASK)),
#         (ValidLength, args(0, 12), kwargs(on_fail=OnFailAction.REFRAIN)),
#     )("Oh Canada")

#     assert response.validation_passed is True
#     assert response.validated_output == "oh canada"


@pytest.mark.skipif(
    not importlib.util.find_spec("transformers")
    and not importlib.util.find_spec("torch"),
    reason="transformers or torch is not installed",
)
def test_hugging_face_model_callable():
    from transformers import AutoModelForCausalLM, AutoTokenizer

    # TODO: Don't actually pull GPT-2 during the test.
    model = AutoModelForCausalLM.from_pretrained("openai-community/gpt2")
    tokenizer = AutoTokenizer.from_pretrained("openai-community/gpt2")

    class Foo(BaseModel):
        bar: str

    g = Guard.from_pydantic(Foo)
    out = g(model.generate, tokenizer=tokenizer, prompt="This is madness.")
    print(out)
    assert False


def test_hugging_face_pipeline_callable():
    from transformers import pipeline

    # TODO: Don't actually pull GPT-2 during the test.
    model = pipeline("text-generation", "openai-community/gpt2")

    class Foo(BaseModel):
        bar: str

    g = Guard.from_pydantic(Foo)
    out = g(model, prompt="This is madness.")
    print(out)
