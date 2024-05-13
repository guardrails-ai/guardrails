import openai
import pytest
from pydantic import BaseModel

from guardrails import AsyncGuard, Rail, Validator
from guardrails.datatypes import verify_metadata_requirements
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
    guard = AsyncGuard.from_rail_string(spec)

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


rail = Rail.from_string_validators([], "empty railspec")
empty_rail_string = """<rail version="0.1">
<output
    type="string"
    description="empty railspec"
/>
</rail>"""


class EmptyModel(BaseModel):
    empty_field: str


i_guard_none = AsyncGuard(rail)
i_guard_two = AsyncGuard(rail, 2)
r_guard_none = AsyncGuard.from_rail("tests/unit_tests/test_assets/empty.rail")
r_guard_two = AsyncGuard.from_rail("tests/unit_tests/test_assets/empty.rail", 2)
rs_guard_none = AsyncGuard.from_rail_string(empty_rail_string)
rs_guard_two = AsyncGuard.from_rail_string(empty_rail_string, 2)
py_guard_none = AsyncGuard.from_pydantic(output_class=EmptyModel)
py_guard_two = AsyncGuard.from_pydantic(output_class=EmptyModel, num_reasks=2)
s_guard_none = AsyncGuard.from_string(validators=[], description="empty railspec")
s_guard_two = AsyncGuard.from_string(
    validators=[], description="empty railspec", num_reasks=2
)


@pytest.mark.parametrize(
    "guard,expected_num_reasks,config_num_reasks",
    [
        (i_guard_none, 1, None),
        (i_guard_two, 2, None),
        (i_guard_none, 3, 3),
        (r_guard_none, 1, None),
        (r_guard_two, 2, None),
        (r_guard_none, 3, 3),
        (rs_guard_none, 1, None),
        (rs_guard_two, 2, None),
        (rs_guard_none, 3, 3),
        (py_guard_none, 1, None),
        (py_guard_two, 2, None),
        (py_guard_none, 3, 3),
        (s_guard_none, 1, None),
        (s_guard_two, 2, None),
        (s_guard_none, 3, 3),
    ],
)
def test_configure(guard: AsyncGuard, expected_num_reasks: int, config_num_reasks: int):
    guard.configure(config_num_reasks)
    assert guard.num_reasks == expected_num_reasks


def guard_init_from_rail():
    guard = AsyncGuard.from_rail("tests/unit_tests/test_assets/simple.rail")
    assert (
        guard.instructions.format().source.strip()
        == "You are a helpful bot, who answers only with valid JSON"
    )
    assert guard.prompt.format().source.strip() == "Extract a string from the text"


def test_use():
    guard: AsyncGuard = (
        AsyncGuard()
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

    # Raises error when trying to `use` a validator on a non-string
    with pytest.raises(RuntimeError):

        class TestClass(BaseModel):
            another_field: str

        py_guard = AsyncGuard.from_pydantic(output_class=TestClass)
        py_guard.use(
            EndsWith("a"), OneLine(), LowerCase(), TwoWords(on_fail=OnFailAction.REASK)
        )

    # Use a combination of prompt, instructions, msg_history and output validators
    # Should only have the output validators in the guard,
    # everything else is in the schema
    guard: AsyncGuard = (
        AsyncGuard()
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
    prompt_validators = guard.rail.prompt_schema.root_datatype.validators
    assert len(prompt_validators) == 2
    assert prompt_validators[0].__class__.__name__ == "LowerCase"
    assert prompt_validators[1].__class__.__name__ == "OneLine"

    instructions_validators = guard.rail.instructions_schema.root_datatype.validators
    assert len(instructions_validators) == 1
    assert instructions_validators[0].__class__.__name__ == "UpperCase"

    msg_history_validators = guard.rail.msg_history_schema.root_datatype.validators
    assert len(msg_history_validators) == 1
    assert msg_history_validators[0].__class__.__name__ == "LowerCase"

    # Check guard for output validators
    assert len(guard._validators) == 2  # only 2 output validators, hence 2

    assert isinstance(guard._validators[0], EndsWith)
    assert guard._validators[0]._kwargs["end"] == "a"
    assert (
        guard._validators[0].on_fail_descriptor == OnFailAction.FIX
    )  # bc this is the default

    assert isinstance(guard._validators[1], TwoWords)
    assert guard._validators[1].on_fail_descriptor == OnFailAction.REASK  # bc we set it

    # Test with an invalid "on" parameter, should raise a ValueError
    with pytest.raises(ValueError):
        guard: AsyncGuard = (
            AsyncGuard()
            .use(EndsWith("a"), on="response")  # invalid on parameter
            .use(OneLine, on="prompt")  # valid on parameter
        )


def test_use_many_instances():
    guard: AsyncGuard = AsyncGuard().use_many(
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

    # Raises error when trying to `use_many` a validator on a non-string
    with pytest.raises(RuntimeError):

        class TestClass(BaseModel):
            another_field: str

        py_guard = AsyncGuard.from_pydantic(output_class=TestClass)
        py_guard.use_many(
            [
                EndsWith("a"),
                OneLine(),
                LowerCase(),
                TwoWords(on_fail=OnFailAction.REASK),
            ]
        )

    # Test with explicitly setting the "on" parameter = "output"
    guard: AsyncGuard = AsyncGuard().use_many(
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
    guard: AsyncGuard = AsyncGuard().use_many(
        OneLine(), LowerCase(), TwoWords(on_fail=OnFailAction.REASK), on="prompt"
    )

    prompt_validators = guard.rail.prompt_schema.root_datatype.validators
    assert len(prompt_validators) == 3
    assert prompt_validators[0].__class__.__name__ == "OneLine"
    assert prompt_validators[1].__class__.__name__ == "LowerCase"
    assert prompt_validators[2].__class__.__name__ == "TwoWords"
    assert len(guard._validators) == 0  # no output validators, hence 0

    # Test with explicitly setting the "on" parameter = "instructions"
    guard: AsyncGuard = AsyncGuard().use_many(
        OneLine(), LowerCase(), TwoWords(on_fail=OnFailAction.REASK), on="instructions"
    )

    instructions_validators = guard.rail.instructions_schema.root_datatype.validators
    assert len(instructions_validators) == 3
    assert instructions_validators[0].__class__.__name__ == "OneLine"
    assert instructions_validators[1].__class__.__name__ == "LowerCase"
    assert instructions_validators[2].__class__.__name__ == "TwoWords"
    assert len(guard._validators) == 0  # no output validators, hence 0

    # Test with explicitly setting the "on" parameter = "msg_history"
    guard: AsyncGuard = AsyncGuard().use_many(
        OneLine(), LowerCase(), TwoWords(on_fail=OnFailAction.REASK), on="msg_history"
    )

    msg_history_validators = guard.rail.msg_history_schema.root_datatype.validators
    assert len(msg_history_validators) == 3
    assert msg_history_validators[0].__class__.__name__ == "OneLine"
    assert msg_history_validators[1].__class__.__name__ == "LowerCase"
    assert msg_history_validators[2].__class__.__name__ == "TwoWords"
    assert len(guard._validators) == 0  # no output validators, hence 0

    # Test with an invalid "on" parameter, should raise a ValueError
    with pytest.raises(ValueError):
        guard: AsyncGuard = AsyncGuard().use_many(
            EndsWith("a", on_fail=OnFailAction.EXCEPTION), OneLine(), on="response"
        )


def test_use_many_tuple():
    guard: AsyncGuard = AsyncGuard().use_many(
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
    guard: AsyncGuard = AsyncGuard().use_many(
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

    # Test with an invalid "on" parameter, should raise a ValueError
    with pytest.raises(ValueError):
        guard: AsyncGuard = AsyncGuard().use_many(
            (EndsWith, ["a"], {"on_fail": OnFailAction.EXCEPTION}),
            OneLine,
            on="response",
        )


def test_validate():
    guard: AsyncGuard = (
        AsyncGuard()
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
    guard: AsyncGuard = (
        AsyncGuard()
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
    guard: AsyncGuard = (
        AsyncGuard()
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
    prompt_validators = guard.rail.prompt_schema.root_datatype.validators
    assert len(prompt_validators) == 2
    assert prompt_validators[0].__class__.__name__ == "OneLine"
    assert prompt_validators[1].__class__.__name__ == "LowerCase"

    instructions_validators = guard.rail.instructions_schema.root_datatype.validators
    assert len(instructions_validators) == 1
    assert instructions_validators[0].__class__.__name__ == "UpperCase"

    msg_history_validators = guard.rail.msg_history_schema.root_datatype.validators
    assert len(msg_history_validators) == 1
    assert msg_history_validators[0].__class__.__name__ == "LowerCase"

    # Check guard for output validators
    assert len(guard._validators) == 2  # only 2 output validators, hence 2

    assert isinstance(guard._validators[0], TwoWords)
    assert guard._validators[0].on_fail_descriptor == OnFailAction.REASK  # bc we set it

    assert isinstance(guard._validators[1], ValidLength)
    assert guard._validators[1]._min == 0
    assert guard._validators[1]._kwargs["min"] == 0
    assert guard._validators[1]._max == 12
    assert guard._validators[1]._kwargs["max"] == 12
    assert (
        guard._validators[1].on_fail_descriptor == OnFailAction.REFRAIN
    )  # bc we set it

    # Test with an invalid "on" parameter, should raise a ValueError
    with pytest.raises(ValueError):
        guard: AsyncGuard = (
            AsyncGuard()
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
#     response = AsyncGuard().use_many(
#         ReadingTime(five_seconds, on_fail=OnFailAction.EXCEPTION),
#         OneLine,
#         (EndsWith, ["a"], {"on_fail": OnFailAction.EXCEPTION}),
#         (LowerCase, kwargs(on_fail=OnFailAction.FIX_REASK, some_other_kwarg="kwarg")),
#         (TwoWords, on_fail(OnFailAction.REASK)),
#         (ValidLength, args(0, 12), kwargs(on_fail=OnFailAction.REFRAIN)),
#     )("Oh Canada")

#     assert response.validation_passed is True
#     assert response.validated_output == "oh canada"
