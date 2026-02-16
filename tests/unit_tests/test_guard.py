from unittest.mock import patch
import pytest

import openai  # noqa: F401
from pydantic import BaseModel

from guardrails import Guard, Validator, register_validator
from guardrails.classes.validation.validation_result import PassResult
from guardrails.utils.validator_utils import verify_metadata_requirements
from guardrails.utils import args, kwargs, on_fail
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
        .use(LowerCase, on="messages")
        .use(OneLine, on="messages")
        .use(
            EndsWith, end="a", on="output"
        )  # default on="output", still explicitly set
        .use(
            TwoWords, on_fail=OnFailAction.REASK
        )  # default on="output", implicitly set
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
        guard._validators[1].on_fail_descriptor == OnFailAction.EXCEPTION
    )  # bc this is the default

    assert isinstance(guard._validators[2], LowerCase)
    assert (
        guard._validators[2].on_fail_descriptor == OnFailAction.EXCEPTION
    )  # bc this is the default

    assert isinstance(guard._validators[3], TwoWords)
    assert guard._validators[3].on_fail_descriptor == OnFailAction.REASK  # bc we set it

    # No longer a constraint
    # # Raises error when trying to `use_many` a validator on a non-string
    # with pytest.raises(RuntimeError):

    class TestClass(BaseModel):
        another_field: str

    py_guard = Guard.for_pydantic(output_class=TestClass)
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
        guard._validators[1].on_fail_descriptor == OnFailAction.EXCEPTION
    )  # bc this is the default

    assert isinstance(guard._validators[2], LowerCase)
    assert (
        guard._validators[2].on_fail_descriptor == OnFailAction.EXCEPTION
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
        guard._validators[0].on_fail_descriptor == OnFailAction.EXCEPTION
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
        guard._validators[1].on_fail_descriptor == OnFailAction.EXCEPTION
    )  # bc this is the default

    # Test with an unrecognized "on" parameter, should warn with a UserWarning
    with pytest.warns(UserWarning):
        guard: Guard = Guard().use_many(
            (EndsWith, ["a"], {"on_fail": OnFailAction.EXCEPTION}),
            OneLine,
            on="response",
        )


# TODO: Move to integration tests; these are not unit tests...
class TestValidate:
    def test_output_only_success(self):
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

    def test_output_only_failure(self):
        guard: Guard = (
            Guard()
            .use(OneLine, on_fail=OnFailAction.NOOP)
            .use(
                LowerCase(on_fail=OnFailAction.FIX), on="output"
            )  # default on="output", still explicitly set
            .use(TwoWords, on_fail=OnFailAction.NOOP)
            .use(ValidLength, 0, 12, on_fail=OnFailAction.REFRAIN)
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
            .use(OneLine, on="prompt")
            .use(LowerCase, on="instructions")
            .use(UpperCase, on="msg_history")
            .use(LowerCase, on="output", on_fail=OnFailAction.FIX)
            .use(TwoWords)
            .use(ValidLength, 0, 12, on_fail=OnFailAction.REFRAIN)
        )

        llm_output: str = "Oh Canada"  # bc it meets our criteria

        response = guard.validate(llm_output)

        assert response.validation_passed is True
        assert response.validated_output == llm_output.lower()

    def test_on_many_failure(self):
        guard: Guard = (
            Guard()
            .use(OneLine, on="messages")
            .use(LowerCase, on="output", on_fail=OnFailAction.FIX)
            .use(TwoWords, on_fail=OnFailAction.NOOP)
            .use(ValidLength, 0, 12, on_fail=OnFailAction.REFRAIN)
        )

        llm_output = "Star Spangled Banner"  # to stick with the theme

        response = guard.validate(llm_output)

        assert response.validation_passed is False
        assert response.validated_output is None


def test_use_and_use_many():
    guard: Guard = (
        Guard()
        .use_many(OneLine(), LowerCase(), on="messages")
        .use_many(
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
            .use_many(OneLine(), LowerCase(), on="messages")
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


class TestGuardSerialization:
    """Test Guard serialization and deserialization."""

    def test_to_dict_basic(self):
        """Test basic to_dict functionality."""
        guard = Guard()
        guard_dict = guard.to_dict()

        assert isinstance(guard_dict, dict)
        assert "id" in guard_dict
        assert "name" in guard_dict
        assert "validators" in guard_dict
        assert "output_schema" in guard_dict

    def test_to_dict_with_validators(self):
        """Test to_dict with validators."""
        guard = Guard().use(LowerCase()).use(OneLine())
        guard_dict = guard.to_dict()

        assert isinstance(guard_dict, dict)
        assert len(guard_dict["validators"]) == 2

    def test_from_dict_none(self):
        """Test from_dict with None input."""
        result = Guard.from_dict(None)
        assert result is None

    def test_from_dict_basic(self):
        """Test basic from_dict functionality."""
        original_guard = Guard()
        guard_dict = original_guard.to_dict()

        restored_guard = Guard.from_dict(guard_dict)

        assert restored_guard is not None
        assert restored_guard.id == original_guard.id
        assert restored_guard.name == original_guard.name

    def test_from_dict_with_validators(self):
        """Test from_dict with validators."""
        original_guard = Guard().use(LowerCase())
        guard_dict = original_guard.to_dict()

        restored_guard = Guard.from_dict(guard_dict)

        assert restored_guard is not None
        assert len(restored_guard.validators) == len(original_guard.validators)

    def test_round_trip_serialization(self):
        """Test that serialization round-trip preserves guard state."""
        original_guard = (
            Guard(name="test-guard", description="Test guard for serialization")
            .use(LowerCase())
            .use(OneLine())
        )

        # Serialize
        guard_dict = original_guard.to_dict()

        # Deserialize
        restored_guard = Guard.from_dict(guard_dict)

        # Compare
        assert restored_guard is not None
        assert restored_guard.name == original_guard.name
        assert restored_guard.description == original_guard.description
        assert len(restored_guard.validators) == len(original_guard.validators)


class TestFetchGuard:
    """Test Guard.fetch_guard functionality."""

    def test_fetch_guard_without_name(self):
        """Test that fetch_guard raises ValueError when name is not
        specified."""
        with pytest.raises(ValueError, match="Name must be specified to fetch a guard"):
            Guard.fetch_guard()

    def test_fetch_guard_with_name(self, mocker):
        """Test fetch_guard with a valid name."""
        # Create a real Guard to return from the mock
        mock_fetched_guard = Guard()
        mock_fetched_guard.name = "fetched-guard"

        # Mock the API client
        mock_api_client = mocker.Mock()
        mock_api_client.fetch_guard.return_value = mock_fetched_guard

        # Mock GuardrailsApiClient constructor
        mocker.patch(
            "guardrails.guard.GuardrailsApiClient", return_value=mock_api_client
        )

        # Mock settings to enable server mode
        mock_settings = mocker.patch("guardrails.guard.settings")
        mock_settings.use_server = True

        result = Guard.fetch_guard(name="test-guard")

        # Should return a Guard instance
        assert isinstance(result, Guard)
        assert result.name == "test-guard"

        # Should have called the API client's fetch_guard
        mock_api_client.fetch_guard.assert_called()

    def test_fetch_guard_with_api_key_and_base_url(self, mocker):
        """Test fetch_guard with custom api_key and base_url."""
        # Create a real Guard to return from the mock
        mock_fetched_guard = Guard()

        mock_api_client = mocker.Mock()
        mock_api_client.fetch_guard.return_value = mock_fetched_guard

        mock_client_class = mocker.patch(
            "guardrails.guard.GuardrailsApiClient", return_value=mock_api_client
        )
        mock_settings = mocker.patch("guardrails.guard.settings")
        mock_settings.use_server = True

        result = Guard.fetch_guard(
            name="test-guard", api_key="test-api-key", base_url="https://test.api.com"
        )

        # Should create a Guard with the specified credentials
        assert isinstance(result, Guard)
        assert result._api_key == "test-api-key"
        assert result._base_url == "https://test.api.com"

        # Should have called GuardrailsApiClient with correct parameters
        # Note: It gets called multiple times during Guard initialization
        assert mock_client_class.call_count >= 1
        mock_client_class.assert_any_call(
            api_key="test-api-key", base_url="https://test.api.com"
        )

    def test_fetch_guard_not_found(self, mocker):
        """Test fetch_guard when guard is not found on server."""
        mock_api_client = mocker.Mock()
        mock_api_client.fetch_guard.return_value = None

        mocker.patch(
            "guardrails.guard.GuardrailsApiClient", return_value=mock_api_client
        )
        mocker.patch("guardrails.guard.settings")

        with pytest.raises(ValueError, match="Guard with name test-guard not found"):
            Guard.fetch_guard(name="test-guard")


class TestErrorSpansInOutput:
    """Test error_spans_in_output functionality."""

    def test_error_spans_no_history(self):
        """Test error_spans_in_output when there is no history."""
        guard = Guard()
        error_spans = guard.error_spans_in_output()

        assert isinstance(error_spans, list)
        assert len(error_spans) == 0

    def test_error_spans_with_empty_history(self):
        """Test error_spans_in_output with empty history."""
        guard = Guard()
        # History is initialized but empty
        error_spans = guard.error_spans_in_output()

        assert isinstance(error_spans, list)
        assert len(error_spans) == 0

    def test_error_spans_with_validation_errors(self, mocker):
        """Test error_spans_in_output with actual validation errors."""
        from guardrails.classes.history import Iteration
        from guardrails.classes.validation.validation_result import ErrorSpan

        guard = Guard()

        # Create a mock call with iterations
        mock_call = mocker.Mock()
        mock_iteration = mocker.Mock(spec=Iteration)

        # Create mock error spans
        error_span1 = ErrorSpan(start=0, end=5, reason="Test error 1")
        error_span2 = ErrorSpan(start=10, end=15, reason="Test error 2")
        mock_iteration.error_spans_in_output = [error_span1, error_span2]

        # Set up the mock structure
        mock_call.iterations = mocker.Mock()
        mock_call.iterations.last = mock_iteration

        # Add to guard history
        guard.history.push(mock_call)

        error_spans = guard.error_spans_in_output()

        assert len(error_spans) == 2
        assert error_spans[0] == error_span1
        assert error_spans[1] == error_span2

    def test_error_spans_handles_attribute_error(self, mocker):
        """Test that error_spans_in_output handles AttributeError
        gracefully."""
        guard = Guard()

        # Create a mock call that raises AttributeError
        mock_call = mocker.Mock()
        mock_call.iterations = []

        guard.history.push(mock_call)

        error_spans = guard.error_spans_in_output()

        assert isinstance(error_spans, list)
        assert len(error_spans) == 0


class TestResponseFormatJsonSchema:
    """Test response_format_json_schema functionality."""

    def test_response_format_json_schema_with_pydantic(self):
        """Test response_format_json_schema with Pydantic model."""

        class TestModel(BaseModel):
            field1: str
            field2: int

        guard = Guard.for_pydantic(output_class=TestModel)
        result = guard.response_format_json_schema()

        assert isinstance(result, dict)
        assert "type" in result
        result_json_schema = result["json_schema"]
        assert "strict" in result_json_schema


class TestUpsertGuard:
    """Test upsert_guard functionality."""

    @patch("guardrails.guard.GuardrailsApiClient")
    def test_upsert_guard(self, mock_api_client):
        """Test upsert_guard saves guard to server."""
        guard = Guard(name="test-guard", use_server=True, preloaded=True)

        guard.upsert_guard()

        assert guard._api_client.upsert_guard.call_count == 1


class TestConfigureExtended:
    """Extended tests for configure method."""

    def test_configure_with_allow_metrics_collection(self, mocker):
        """Test configure with allow_metrics_collection parameter."""
        guard = Guard()

        # Mock the hub telemetry configuration
        mock_configure = mocker.patch.object(guard, "_configure_hub_telemtry")

        guard.configure(allow_metrics_collection=True)

        mock_configure.assert_called_once()
        mock_configure.assert_called_with(True)

    def test_configure_with_both_parameters(self, mocker):
        """Test configure with both num_reasks and allow_metrics_collection."""
        guard = Guard()

        mock_configure = mocker.patch.object(guard, "_configure_hub_telemtry")

        guard.configure(num_reasks=3, allow_metrics_collection=False)

        assert guard._num_reasks == 3
        assert guard._allow_metrics_collection is False
        mock_configure.assert_called_once()

    def test_configure_multiple_times(self):
        """Test that configure can be called multiple times."""
        guard = Guard()

        guard.configure(num_reasks=1)
        assert guard._num_reasks == 1

        guard.configure(num_reasks=5)
        assert guard._num_reasks == 5

    def test_configure_with_none_resets(self, mocker):
        """Test that configure with None resets num_reasks."""
        guard = Guard()

        mock__set_num_reasks = mocker.patch.object(guard, "_set_num_reasks")

        guard.configure(num_reasks=3)
        mock__set_num_reasks.assert_called_once()
        mock__set_num_reasks.assert_called_with(3)

        mock__set_num_reasks.reset_mock()

        guard.configure(num_reasks=None)
        mock__set_num_reasks.assert_not_called()


class TestHistoryManagement:
    """Test Guard history management."""

    def test_history_initialization(self):
        """Test that history is properly initialized."""
        guard = Guard()

        assert hasattr(guard, "history")
        assert guard.history is not None
        assert len(guard.history) == 0

    def test_history_max_length_default(self):
        """Test that history has default max length."""
        guard = Guard()

        assert guard._history_max_length == 10

    def test_history_max_length_custom(self):
        """Test history with custom max length."""
        guard = Guard(history_max_length=5)

        assert guard._history_max_length == 5

    def test_history_respects_max_length(self, mocker):
        """Test that history respects max length."""
        guard = Guard(history_max_length=2)

        # Create mock calls
        call1 = mocker.Mock()
        call2 = mocker.Mock()
        call3 = mocker.Mock()

        # Add calls to history
        guard.history.push(call1)
        guard.history.push(call2)
        guard.history.push(call3)

        # History should only keep the last 2
        assert len(guard.history) == 2


class TestGuardProperties:
    """Test Guard property access."""

    def test_id_property(self):
        """Test id property access."""
        guard = Guard()

        assert hasattr(guard, "id")
        assert guard.id is not None
        assert isinstance(guard.id, str)

    def test_name_property(self):
        """Test name property access."""
        guard = Guard(name="test-guard")

        assert guard.name == "test-guard"

    def test_description_property(self):
        """Test description property access."""
        guard = Guard(description="Test description")

        assert guard.description == "Test description"

    def test_validators_property(self):
        """Test validators property access."""
        guard = Guard().use(LowerCase())

        assert hasattr(guard, "validators")
        assert isinstance(guard.validators, list)
        assert len(guard.validators) == 1

    def test_output_schema_property(self):
        """Test output_schema property access."""
        guard = Guard()

        assert hasattr(guard, "output_schema")
        assert guard.output_schema is not None


class TestGuardInitialization:
    """Test various Guard initialization scenarios."""

    def test_init_with_all_parameters(self):
        """Test Guard initialization with all parameters."""
        validators = []
        output_schema = {"type": "string", "description": "test"}

        guard = Guard(
            id="test-id",
            name="test-guard",
            description="Test guard",
            validators=validators,
            output_schema=output_schema,
            history_max_length=15,
        )

        assert guard.id == "test-id"
        assert guard.name == "test-guard"
        assert guard.description == "Test guard"
        assert guard.validators == validators
        assert guard._history_max_length == 15

    def test_init_with_minimal_parameters(self):
        """Test Guard initialization with minimal parameters."""
        guard = Guard()

        # Check defaults
        assert guard.id is not None
        assert guard.name is not None
        assert guard.name.startswith("gr-")
        assert guard.validators == []
        assert guard.output_schema is not None
        assert guard._history_max_length == 10

    def test_init_with_custom_id(self):
        """Test that custom ID is preserved."""
        custom_id = "my-custom-id"
        guard = Guard(id=custom_id)

        assert guard.id == custom_id
        assert guard.name == f"gr-{custom_id}"

    @patch("guardrails.guard.GuardrailsApiClient")
    def test_init_with_api_credentials(self, mock_api_client):
        """Test Guard initialization with API credentials."""
        guard = Guard(
            api_key="test-key", base_url="https://test.api.com", use_server=True
        )

        # These are stored as private attributes
        assert guard._api_key == "test-key"
        assert guard._base_url == "https://test.api.com"


class TestJsonFunctionCallingTool:
    """Test json_function_calling_tool functionality."""

    def test_json_function_calling_tool_basic(self):
        """Test basic json_function_calling_tool functionality."""
        guard = Guard()

        # Test with empty tools list
        result = guard.json_function_calling_tool()
        json_tool = result[0]

        assert isinstance(result, list)
        assert isinstance(json_tool, dict)
        assert "type" in json_tool
        assert json_tool["type"] == "function"

    def test_json_function_calling_tool_with_pydantic(self):
        """Test json_function_calling_tool with Pydantic model."""

        class TestModel(BaseModel):
            field1: str
            field2: int

        guard = Guard.for_pydantic(output_class=TestModel)
        result = guard.json_function_calling_tool()
        json_tool = result[0]

        assert isinstance(result, list)
        assert isinstance(json_tool, dict)
        assert "type" in json_tool
        assert "function" in json_tool

    def test_json_function_calling_tool_with_custom_tools(self):
        """Test json_function_calling_tool with custom tools."""
        guard = Guard()

        custom_tools = [{"type": "function", "function": {"name": "test_func"}}]

        result = guard.json_function_calling_tool(tools=custom_tools)

        assert isinstance(result, list)
