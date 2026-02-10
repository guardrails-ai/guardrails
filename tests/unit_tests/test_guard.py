import pytest

import openai  # noqa: F401
from pydantic import BaseModel

from guardrails import Guard, Validator, register_validator
from guardrails.classes.validation.validation_result import PassResult
from guardrails.utils.validator_utils import verify_metadata_requirements
from guardrails.types import OnFailAction
from tests.integration_tests.test_assets.validators import (
    EndsWith,
    LowerCase,
    OneLine,
    TwoWords,
    UpperCase,
    ValidLength,
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
@pytest.mark.skip(reason="Only for OpenAI v0")  # FIXME: Rewrite for OpenAI v1
async def test_required_metadata(spec, metadata, error_message):
    guard = Guard.for_rail_string(spec)

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
r_guard_none = Guard.for_rail("tests/unit_tests/test_assets/empty.rail")
r_guard_two = Guard.for_rail("tests/unit_tests/test_assets/empty.rail")
r_guard_two.configure(num_reasks=2)
rs_guard_none = Guard.for_rail_string(empty_rail_string)
rs_guard_two = Guard.for_rail_string(empty_rail_string)
rs_guard_two.configure(num_reasks=2)
py_guard_none = Guard.for_pydantic(output_class=EmptyModel)
py_guard_two = Guard.for_pydantic(output_class=EmptyModel)
py_guard_two.configure(num_reasks=2)
s_guard_none = Guard.for_string(validators=[], string_description="empty railspec")
s_guard_two = Guard.for_string(validators=[], description="empty railspec")
s_guard_two.configure(num_reasks=2)


class TestConfigure:
    def test_num_reasks(self):
        guard = Guard()
        guard.configure()

        assert guard._num_reasks is None

        guard.configure(num_reasks=2)

        assert guard._num_reasks == 2


def test_use():
    guard: Guard = Guard().use(
        EndsWith("a"),
        OneLine(),
        LowerCase(),
        TwoWords(on_fail=OnFailAction.REASK),
        ValidLength(0, 12, on_fail=OnFailAction.REFRAIN),
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
        guard._validators[1].on_fail_descriptor == OnFailAction.EXCEPTION
    )  # bc this is the default

    assert isinstance(guard._validators[2], LowerCase)
    assert (
        guard._validators[2].on_fail_descriptor == OnFailAction.EXCEPTION
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

    py_guard = Guard.for_pydantic(output_class=TestClass)
    py_guard.use(EndsWith("a"))
    assert py_guard._validator_map.get("$") == [EndsWith("a")]

    # Use a combination of prompt, instructions, msg_history and output validators
    # Should only have the output validators in the guard,
    # everything else is in the schema
    guard: Guard = (
        Guard()
        .use(LowerCase(), OneLine(), on="messages")
        .use(
            EndsWith(end="a"), TwoWords(on_fail=OnFailAction.REASK), on="output"
        )  # default on="output", still explicitly set
    )

    # Check schemas for messages validators
    prompt_validators = guard._validator_map.get("messages", [])
    assert len(prompt_validators) == 2
    assert prompt_validators[0].__class__.__name__ == "LowerCase"
    assert prompt_validators[1].__class__.__name__ == "OneLine"

    # Check guard for output validators
    assert len(guard._validators) == 4

    assert isinstance(guard._validators[2], EndsWith)
    assert guard._validators[2]._kwargs["end"] == "a"
    assert (
        guard._validators[2].on_fail_descriptor == OnFailAction.FIX
    )  # bc this is the default

    assert isinstance(guard._validators[3], TwoWords)
    assert guard._validators[3].on_fail_descriptor == OnFailAction.REASK  # bc we set it

    # Test with an unrecognized "on" parameter, should warn with a UserWarning
    with pytest.warns(UserWarning):
        guard: Guard = (
            Guard()
            .use(EndsWith("a"), on="response")  # invalid on parameter
            .use(OneLine(), on="prompt")  # valid on parameter
        )


# TODO: Move to integration tests; these are not unit tests...
class TestValidate:
    def test_output_only_success(self):
        guard: Guard = Guard().use(
            OneLine(),
            LowerCase(on_fail=OnFailAction.FIX),
            TwoWords(),
            ValidLength(0, 12, on_fail=OnFailAction.REFRAIN),
        )

        llm_output: str = "Oh Canada"  # bc it meets our criteria

        response = guard.validate(llm_output)

        assert response.validation_passed is True
        assert response.validated_output == llm_output.lower()

    def test_output_only_failure(self):
        guard: Guard = Guard().use(
            OneLine(on_fail=OnFailAction.NOOP),
            LowerCase(on_fail=OnFailAction.FIX),
            TwoWords(on_fail=OnFailAction.NOOP),
            ValidLength(0, 12, on_fail=OnFailAction.REFRAIN),
        )

        llm_output = "Star Spangled Banner"  # to stick with the theme

        response = guard.validate(llm_output)

        assert response.validation_passed is False
        assert response.validated_output is None

    def test_on_many_success(self):
        # Test with a combination of prompt, output,
        #   instructions and msg_history validators
        # Should still only use the output validators to validate the output
        guard: Guard = (
            Guard()
            .use(OneLine(), on="prompt")
            .use(LowerCase(), on="instructions")
            .use(UpperCase(), on="msg_history")
            .use(
                LowerCase(on_fail=OnFailAction.FIX),
                TwoWords(),
                ValidLength(0, 12, on_fail=OnFailAction.REFRAIN),
            )
        )

        llm_output: str = "Oh Canada"  # bc it meets our criteria

        response = guard.validate(llm_output)

        assert response.validation_passed is True
        assert response.validated_output == llm_output.lower()

    def test_on_many_failure(self):
        guard: Guard = (
            Guard()
            .use(OneLine(), on="messages")
            .use(
                LowerCase(on_fail=OnFailAction.FIX),
                TwoWords(on_fail=OnFailAction.NOOP),
                ValidLength(0, 12, on_fail=OnFailAction.REFRAIN),
            )
        )

        llm_output = "Star Spangled Banner"  # to stick with the theme

        response = guard.validate(llm_output)

        assert response.validation_passed is False
        assert response.validated_output is None


def test_multi_use():
    guard: Guard = (
        Guard()
        .use(OneLine(), LowerCase(), on="messages")
        .use(
            TwoWords(on_fail=OnFailAction.REASK),
            ValidLength(0, 12, on_fail=OnFailAction.REFRAIN),
            on="output",
        )
    )

    # Check schemas for messages validators
    prompt_validators = guard._validator_map.get("messages", [])
    assert len(prompt_validators) == 2
    assert prompt_validators[0].__class__.__name__ == "OneLine"
    assert prompt_validators[1].__class__.__name__ == "LowerCase"

    # Check guard for validators
    assert len(guard._validators) == 4

    assert isinstance(guard._validators[2], TwoWords)
    assert guard._validators[2].on_fail_descriptor == OnFailAction.REASK  # bc we set it

    assert isinstance(guard._validators[3], ValidLength)
    assert guard._validators[3]._min == 0
    assert guard._validators[3]._kwargs["min"] == 0
    assert guard._validators[3]._max == 12
    assert guard._validators[3]._kwargs["max"] == 12
    assert (
        guard._validators[3].on_fail_descriptor == OnFailAction.REFRAIN
    )  # bc we set it

    # Test with an unrecognized "on" parameter, should warn with a UserWarning
    with pytest.warns(UserWarning):
        guard: Guard = (
            Guard()
            .use(OneLine(), LowerCase(), on="messages")
            .use(
                TwoWords(on_fail=OnFailAction.REASK),
                ValidLength(0, 12, on_fail=OnFailAction.REFRAIN),
                on="response",  # invalid "on" parameter
            )
        )
