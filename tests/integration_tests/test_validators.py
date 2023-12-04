# noqa:W291
import os
from typing import Any, Callable, Dict, Optional, Union

import pytest

from guardrails import Guard, Validator, register_validator
from guardrails.datatypes import DataType
from guardrails.schema import StringSchema
from guardrails.validator_base import PassResult, ValidationResult
from guardrails.validators import DetectSecrets, PIIFilter, SimilarToList

from .mock_embeddings import MOCK_EMBEDDINGS
from .mock_llm_outputs import MockOpenAICallable
from .mock_presidio import MockAnalyzerEngine, MockAnonymizerEngine, mock_anonymize
from .mock_secrets import (
    EXPECTED_SECRETS_CODE_SNIPPET,
    NO_SECRETS_CODE_SNIPPET,
    SECRETS_CODE_SNIPPET,
    MockDetectSecrets,
    mock_get_unique_secrets,
)


def test_similar_to_list():
    """Test initialisation of SimilarToList."""

    int_prev_values = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    str_prev_values = ["broadcom", "paypal"]

    def embed_function(text: str):
        """Mock embedding function."""
        return MOCK_EMBEDDINGS[text]

    # Initialise Guard from string (default parameters)
    guard = Guard.from_string(
        validators=[SimilarToList()],
        description="testmeout",
    )

    guard = Guard.from_string(
        validators=[SimilarToList(standard_deviations=2, threshold=0.2, on_fail="fix")],
        description="testmeout",
    )

    # Check types remain intact
    output_schema: StringSchema = guard.rail.output_schema
    data_type: DataType = output_schema.root_datatype
    validators = data_type.validators_attr.validators
    validator: SimilarToList = validators[0]

    assert isinstance(validator._standard_deviations, int)
    assert isinstance(validator._threshold, float)

    # 1. Test for integer values
    # 1.1 Test for values within the standard deviation
    # llm_output must be a string
    val = "3"
    _, output, *rest = guard.parse(
        llm_output=val,
        metadata={"prev_values": int_prev_values},
    )
    # Guard.from_string will always return a string
    # For other return types, we would need some return_type specifiers
    assert output == str(val)

    # 1.2 Test not passing prev_values
    # Should raise ValueError
    val = "3"
    output = guard.parse(
        llm_output=val,
    )
    assert output.error is not None

    # 1.3 Test passing str prev values for int val
    # Should raise ValueError
    val = "3"
    output = guard.parse(
        llm_output=val,
        metadata={"prev_values": [str(i) for i in int_prev_values]},
    )
    assert output.error is not None

    # 1.4 Test for values outside the standard deviation
    val = "300"
    output = guard.parse(
        llm_output=val,
        metadata={"prev_values": int_prev_values},
    )
    assert output.validated_output is None

    # 2. Test for string values
    # 2.1 Test for values within the standard deviation
    val = "cisco"
    output = guard.parse(
        llm_output=val,
        metadata={"prev_values": str_prev_values, "embed_function": embed_function},
    )
    assert output.validated_output == val

    # 2.2 Test not passing prev_values
    # Should raise ValueError
    val = "cisco"
    output = guard.parse(
        llm_output=val,
        metadata={"embed_function": embed_function},
    )
    assert output.error is not None

    # 2.3 Test passing int prev values for str val
    # Should raise ValueError
    val = "cisco"
    output = guard.parse(
        llm_output=val,
        metadata={"prev_values": int_prev_values, "embed_function": embed_function},
    )
    assert output.error is not None

    # 2.4 Test not pasisng embed_function
    # Should raise ValueError
    val = "cisco"
    output = guard.parse(
        llm_output=val,
        metadata={"prev_values": str_prev_values},
    )
    assert output.error is not None

    # 2.5 Test for values outside the standard deviation
    val = "taj mahal"
    output = guard.parse(
        llm_output=val,
        metadata={"prev_values": str_prev_values, "embed_function": embed_function},
    )
    assert output.validated_output is None


def test_detect_secrets(mocker):
    """Test the DetectSecrets validator."""

    # Set the mockers
    mocker.patch("guardrails.validators.detect_secrets", new=MockDetectSecrets)
    mocker.patch(
        "guardrails.validators.DetectSecrets.get_unique_secrets",
        new=mock_get_unique_secrets,
    )

    # Initialise Guard from string
    guard = Guard.from_string(
        validators=[DetectSecrets(on_fail="fix")],
        description="testmeout",
    )

    # ----------------------------
    # 1. Test with SECRETS_CODE_SNIPPET
    output = guard.parse(
        llm_output=SECRETS_CODE_SNIPPET,
    )
    # Check if the output is different from the input
    assert output.validated_output != SECRETS_CODE_SNIPPET

    # Check if output matches the expected output
    assert output.validated_output == EXPECTED_SECRETS_CODE_SNIPPET

    # Check if temp.txt does not exist in current directory
    assert not os.path.exists("temp.txt")

    # ----------------------------
    # 2. Test with NO_SECRETS_CODE_SNIPPET
    output = guard.parse(
        llm_output=NO_SECRETS_CODE_SNIPPET,
    )
    # Check if the output is same as the input
    assert output.validated_output == NO_SECRETS_CODE_SNIPPET

    # Check if temp.txt does not exist in current directory
    assert not os.path.exists("temp.txt")

    # ----------------------------
    # 3. Test with a non-multi-line string
    # Should raise UserWarning
    with pytest.warns(UserWarning):
        output = guard.parse(
            llm_output="import os",
        )

    # Check if the output is same as the input
    assert output.validated_output == "import os"

    # Check if temp.txt does not exist in current directory
    assert not os.path.exists("temp.txt")


def test_pii_filter(mocker):
    """Integration test for PIIFilter."""

    # Mock the the intialisations of AnalyzerEngine and AnonymizerEngine
    mocker.patch("guardrails.validators.AnalyzerEngine", new=MockAnalyzerEngine)
    mocker.patch("guardrails.validators.AnonymizerEngine", new=MockAnonymizerEngine)

    # Mock the analyze and anomymize functions
    mocker.patch(
        "guardrails.validators.PIIFilter.get_anonymized_text", new=mock_anonymize
    )

    # ------------------
    # 1. Initialise Guard from string with setting pii_entities as a string
    # Also check whether all parameters are correctly initialised
    guard = Guard.from_string(
        validators=[PIIFilter(pii_entities="pii", on_fail="fix")],
        description="testmeout",
    )

    # Do parse call
    text = "My email address is demo@lol.com, and my phone number is 1234567890"
    output = guard.parse(
        llm_output=text,
    )
    # Validated output should be different from input
    assert output.validated_output != text

    # Validated output should contain masked pii entities
    assert all(
        entity in output.validated_output
        for entity in ["<EMAIL_ADDRESS>", "<PHONE_NUMBER>"]
    )

    # ------------------
    # 2. Initialise Guard from string with setting pii_entities as a list
    # Also check whether all parameters are correctly initialised
    guard = Guard.from_string(
        validators=[
            PIIFilter(pii_entities=["EMAIL_ADDRESS", "PHONE_NUMBER"], on_fail="fix")
        ],
        description="testmeout",
    )

    # Do parse call
    text = "My email address is demo@lol.com, and my phone number is 1234567890"
    output = guard.parse(
        llm_output=text,
    )
    # Validated output should be different from input
    assert output.validated_output != text

    # Validated output should contain masked pii entities
    assert all(
        entity in output.validated_output
        for entity in ["<EMAIL_ADDRESS>", "<PHONE_NUMBER>"]
    )

    # Check with text without any pii entities
    text = "My email address is xyz and my phone number is unavailable."
    output = guard.parse(
        llm_output=text,
    )
    # Validated output should be same as input
    assert output.validated_output == text

    # ------------------
    # 3. Initialise Guard from string without setting pii_entities
    # Also don't pass through metadata
    # Should raise ValueError
    guard = Guard.from_string(
        validators=[PIIFilter(on_fail="fix")],
        description="testmeout",
    )

    text = "My email address is demo@lol.com, and my phone number is 1234567890"
    output = guard.parse(
        llm_output=text,
    )
    assert output.error is not None

    # ------------------
    # 4. Initialise Guard from string without setting pii_entities
    guard = Guard.from_string(
        validators=[PIIFilter(on_fail="fix")],
        description="testmeout",
    )
    text = "My email address is demo@lol.com, and my phone number is 1234567890"

    # Now try with string of pii entities passed through metadata
    output = guard.parse(
        llm_output=text,
        metadata={"pii_entities": "pii"},
    )
    # Validated output should be different from input
    assert output.validated_output != text

    # Validated output should contain masked pii entities
    assert all(
        entity in output.validated_output
        for entity in ["<EMAIL_ADDRESS>", "<PHONE_NUMBER>"]
    )

    # Now try with list of pii entities passed through metadata
    output = guard.parse(
        llm_output=text,
        metadata={"pii_entities": ["EMAIL_ADDRESS", "PHONE_NUMBER"]},
    )
    # Validated output should be different from input
    assert output.validated_output != text

    # Validated output should contain masked pii entities
    assert all(
        entity in output.validated_output
        for entity in ["<EMAIL_ADDRESS>", "<PHONE_NUMBER>"]
    )

    # ------------------
    # 5. Initialise Guard from string setting
    # pii_entities as a string "pii" -> all entities
    # But also pass in metadata with all pii_entities as a list
    # only containing EMAIL_ADDRESS
    # metadata should override the pii_entities passed in the constructor,
    # and only mask in EMAIL_ADDRESS

    guard = Guard.from_string(
        validators=[PIIFilter(pii_entities="pii", on_fail="fix")],
        description="testmeout",
    )
    text = "My email address is demo@lol.com, and my phone number is 1234567890"

    output = guard.parse(
        llm_output=text,
        metadata={"pii_entities": ["EMAIL_ADDRESS"]},
    )
    # Validated output should be different from input
    assert output.validated_output != text

    # Validated output should contain masked EMAIL_ADDRESS
    # and not PHONE_NUMBER
    assert "<EMAIL_ADDRESS>" in output.validated_output
    assert "<PHONE_NUMBER>" not in output.validated_output

    # ------------------
    # 6. Initialise Guard from string setting an incorrect string of pii_entities
    # Should raise ValueError during validate

    guard = Guard.from_string(
        validators=[PIIFilter(pii_entities="piii", on_fail="fix")],
        description="testmeout",
    )
    text = "My email address is demo@lol.com, and my phone number is 1234567890"

    output = guard.parse(
        llm_output=text,
    )
    assert output.error is not None


@register_validator("mycustominstancecheckvalidator", data_type="string")
class MyValidator(Validator):
    def __init__(
        self,
        an_instance_attr: str,
        on_fail: Optional[Union[Callable, str]] = None,
        **kwargs
    ):
        self.an_instance_attr = an_instance_attr
        super().__init__(on_fail=on_fail, an_instance_attr=an_instance_attr, **kwargs)

    def validate(self, value: Any, metadata: Dict[str, Any]) -> ValidationResult:
        return PassResult()


@pytest.mark.parametrize(
    "instance_attr",
    [
        "a",
        object(),
    ],
)
def test_validator_instance_attr_equality(mocker, instance_attr):
    mocker.patch("guardrails.llm_providers.OpenAICallable", new=MockOpenAICallable)

    validator = MyValidator(an_instance_attr=instance_attr)

    assert validator.an_instance_attr is instance_attr

    guard = Guard.from_string(
        validators=[validator],
        prompt="",
    )

    assert (
        guard.rail.output_schema.root_datatype.validators[0].an_instance_attr
        == instance_attr
    )
