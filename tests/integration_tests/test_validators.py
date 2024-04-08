# noqa:W291
import os
from typing import Any, Callable, Dict, Optional, Union

import pytest

from guardrails import Guard, Validator, register_validator
from guardrails.datatypes import DataType
from guardrails.schema import StringSchema
from guardrails.validator_base import OnFailAction, PassResult, ValidationResult
from guardrails.validators import (
    DetectSecrets,
    IsHighQualityTranslation,
    PIIFilter,
    SimilarToList,
    ToxicLanguage,
)

from ..unit_tests.mocks.mock_comet import BAD_TRANSLATION, GOOD_TRANSLATION, MockModel
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
from .mock_toxic_language import (
    EXPECTED_PARAGRAPH_WITH_TOXIC_SENTENCES,
    NON_TOXIC_PARAGRAPH,
    PARAGRAPH_WITH_TOXIC_SENTENCES,
    TOXIC_PARAGRAPH,
    MockPipeline,
    mock_get_toxicity,
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
        validators=[
            SimilarToList(
                standard_deviations=2, threshold=0.2, on_fail=OnFailAction.FIX
            )
        ],
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
    with pytest.raises(ValueError) as excinfo:
        guard.parse(
            llm_output=val,
        )
    assert (
        str(excinfo.value) == "You must provide a list of previous values in metadata."
    )

    # 1.3 Test passing str prev values for int val
    # Should raise ValueError
    val = "3"
    with pytest.raises(ValueError) as excinfo:
        guard.parse(
            llm_output=val,
            metadata={"prev_values": [str(i) for i in int_prev_values]},
        )
    assert str(excinfo.value) == (
        "Both given value and all the previous values must be "
        "integers in order to use the distribution check validator."
    )

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
    with pytest.raises(ValueError) as excinfo:
        guard.parse(
            llm_output=val,
            metadata={"embed_function": embed_function},
        )
    assert (
        str(excinfo.value) == "You must provide a list of previous values in metadata."
    )

    # 2.3 Test passing int prev values for str val
    # Should raise ValueError
    val = "cisco"
    with pytest.raises(ValueError) as excinfo:
        guard.parse(
            llm_output=val,
            metadata={"prev_values": int_prev_values, "embed_function": embed_function},
        )
    assert str(excinfo.value) == (
        "Both given value and all the previous values must be "
        "strings in order to use the distribution check validator."
    )

    # 2.4 Test not pasisng embed_function
    # Should raise ValueError
    val = "cisco"
    with pytest.raises(ValueError) as excinfo:
        guard.parse(
            llm_output=val,
            metadata={"prev_values": str_prev_values},
        )
    assert str(excinfo.value) == (
        "You must provide `embed_function` in metadata in order to "
        "check the semantic similarity of the generated string."
    )

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
        validators=[DetectSecrets(on_fail=OnFailAction.FIX)],
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
    mocker.patch(
        "guardrails.validators.pii_filter.AnalyzerEngine", new=MockAnalyzerEngine
    )
    mocker.patch(
        "guardrails.validators.pii_filter.AnonymizerEngine", new=MockAnonymizerEngine
    )

    # Mock the analyze and anomymize functions
    mocker.patch(
        "guardrails.validators.PIIFilter.get_anonymized_text", new=mock_anonymize
    )

    # ------------------
    # 1. Initialise Guard from string with setting pii_entities as a string
    # Also check whether all parameters are correctly initialised
    guard = Guard.from_string(
        validators=[PIIFilter(pii_entities="pii", on_fail=OnFailAction.FIX)],
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
            PIIFilter(
                pii_entities=["EMAIL_ADDRESS", "PHONE_NUMBER"], on_fail=OnFailAction.FIX
            )
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
        validators=[PIIFilter(on_fail=OnFailAction.FIX)],
        description="testmeout",
    )

    text = "My email address is demo@lol.com, and my phone number is 1234567890"
    with pytest.raises(ValueError) as excinfo:
        guard.parse(
            llm_output=text,
        )
    assert str(excinfo.value) == (
        "`pii_entities` must be set in order to use the `PIIFilter` validator."
        "Add this: `pii_entities=['PERSON', 'PHONE_NUMBER']`"
        "OR pii_entities='pii' or 'spi'"
        "in init or metadata."
    )

    # ------------------
    # 4. Initialise Guard from string without setting pii_entities
    guard = Guard.from_string(
        validators=[PIIFilter(on_fail=OnFailAction.FIX)],
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
        validators=[PIIFilter(pii_entities="pii", on_fail=OnFailAction.FIX)],
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
        validators=[PIIFilter(pii_entities="piii", on_fail=OnFailAction.FIX)],
        description="testmeout",
    )
    text = "My email address is demo@lol.com, and my phone number is 1234567890"

    with pytest.raises(ValueError) as excinfo:
        guard.parse(
            llm_output=text,
        )
    assert str(excinfo.value) == "`pii_entities` must be one of ['pii', 'spi']"


def test_toxic_language(mocker):
    """Test the integration of the ToxicLanguage validator.

    1. Test default initialisation (should be validation_method="sentence"
     and threshold=0.5)
    2. Test with a toxic paragraph (with validation_method="full")
    3. Test with a paragraph containing toxic sentences
     (with validation_method="sentence")
    4. Text with a non-toxic paragraph (with validation_method="full")
    5. Test with a paragraph containing no toxic sentences
     (with validation_method="sentence")
    6. Test with a paragraph also specifying threshold
    """

    # Set the mockers
    mocker.patch("guardrails.validators.toxic_language.pipeline", new=MockPipeline)
    mocker.patch(
        "guardrails.validators.toxic_language.ToxicLanguage.get_toxicity",
        new=mock_get_toxicity,
    )

    # ----------------------------
    # 1. Test default initialisation (should be validation_method="sentence"
    # and threshold=0.25)
    guard = Guard.from_string(
        validators=[ToxicLanguage(on_fail=OnFailAction.FIX)],
        description="testmeout",
    )

    # ----------------------------
    # 2. Test with a toxic paragraph (with validation_method="full")
    # Should return empty string
    guard = Guard.from_string(
        validators=[ToxicLanguage(validation_method="full", on_fail=OnFailAction.FIX)],
        description="testmeout",
    )

    output = guard.parse(
        llm_output=TOXIC_PARAGRAPH,
    ).validated_output
    # Check if the output is empty
    assert output == ""

    # ----------------------------
    # 3. Test with a paragraph containing toxic sentences
    # (with validation_method="sentence")
    # Should return a paragraph with toxic sentences removed
    guard = Guard.from_string(
        validators=[
            ToxicLanguage(validation_method="sentence", on_fail=OnFailAction.FIX)
        ],
        description="testmeout",
    )

    output = guard.parse(
        llm_output=PARAGRAPH_WITH_TOXIC_SENTENCES,
    ).validated_output

    # Check if the output matches the expected output
    assert output == EXPECTED_PARAGRAPH_WITH_TOXIC_SENTENCES

    # ----------------------------
    # 4. Text with a non-toxic paragraph (with validation_method="full")
    # Should return the same paragraph
    guard = Guard.from_string(
        validators=[ToxicLanguage(validation_method="full", on_fail=OnFailAction.FIX)],
        description="testmeout",
    )

    output = guard.parse(
        llm_output=NON_TOXIC_PARAGRAPH,
    ).validated_output
    # Check if the output is same as the input
    assert output == NON_TOXIC_PARAGRAPH

    # ----------------------------
    # 5. Test with a paragraph containing no toxic sentences
    # (with validation_method="sentence")
    # Should return the same paragraph

    guard = Guard.from_string(
        validators=[
            ToxicLanguage(validation_method="sentence", on_fail=OnFailAction.FIX)
        ],
        description="testmeout",
    )

    output = guard.parse(
        llm_output=NON_TOXIC_PARAGRAPH,
    ).validated_output
    # Check if the output is same as the input
    assert output == NON_TOXIC_PARAGRAPH

    # ----------------------------
    # 6. Test with a paragraph also specifying threshold
    # Should return a paragraph with toxic sentences removed
    guard = Guard.from_string(
        validators=[
            ToxicLanguage(
                validation_method="sentence", threshold=0.1, on_fail=OnFailAction.FIX
            )
        ],
        description="testmeout",
    )

    output = guard.parse(
        llm_output=NON_TOXIC_PARAGRAPH,
    ).validated_output
    # Check if the output matches the expected output
    assert output == NON_TOXIC_PARAGRAPH


def test_translation_quality(mocker):
    # Set the mockers
    mocker.patch(
        "guardrails.validators.is_high_quality_translation.download_model",
        return_value="some_path",
    )
    mocker.patch(
        "guardrails.validators.is_high_quality_translation.load_from_checkpoint",
        return_value=MockModel(),
    )

    # ----------------------------
    # 1. Test with a good translation
    # Should return the same translation
    guard = Guard.from_string(
        validators=[IsHighQualityTranslation(on_fail=OnFailAction.FIX)],
        description="testmeout",
    )

    output = guard.parse(
        llm_output=GOOD_TRANSLATION,
        metadata={"translation_source": "some input"},
    ).validated_output

    # Check if the output is same as the input
    assert output == GOOD_TRANSLATION

    # ----------------------------

    # 2. Test with a bad translation
    # Should return None
    guard = Guard.from_string(
        validators=[IsHighQualityTranslation(on_fail=OnFailAction.FIX)],
        description="testmeout",
    )

    output = guard.parse(
        llm_output=BAD_TRANSLATION,
        metadata={"translation_source": "some input"},
    ).validated_output

    # Check if the output is empty
    assert output == ""


@register_validator("mycustominstancecheckvalidator", data_type="string")
class MyValidator(Validator):
    def __init__(
        self,
        an_instance_attr: str,
        on_fail: Optional[Union[Callable, str]] = None,
        **kwargs,
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


@pytest.mark.parametrize(
    "output,throws,error_message",
    [
        ("Ice cream is frozen.", False, ""),
        (
            "Ice cream is a frozen dairy product that is consumed in many places.",
            True,
            "String should be readable within 0.05 minutes.",
        ),
        ("This response isn't relevant.", True, "Result must match Ice cream"),
    ],
)
def test_validators_as_runnables(output: str, throws: bool, error_message: str):
    from langchain_core.language_models import LanguageModelInput
    from langchain_core.messages import AIMessage, BaseMessage
    from langchain_core.output_parsers import StrOutputParser
    from langchain_core.prompts import ChatPromptTemplate
    from langchain_core.runnables import Runnable, RunnableConfig

    from guardrails.errors import ValidationError
    from guardrails.validators import ReadingTime, RegexMatch

    class MockModel(Runnable):
        def invoke(
            self, input: LanguageModelInput, config: Optional[RunnableConfig] = None
        ) -> BaseMessage:
            return AIMessage(content=output)

    prompt = ChatPromptTemplate.from_template("ELIF: {topic}")
    model = MockModel()
    regex_match = RegexMatch("Ice cream", match_type="search")
    reading_time = ReadingTime(0.05)  # 3 seconds
    output_parser = StrOutputParser()

    chain = prompt | model | regex_match | reading_time | output_parser

    topic = "ice cream"
    if throws:
        with pytest.raises(ValidationError) as exc_info:
            chain.invoke({"topic": topic})

        assert str(exc_info.value) == (
            "The response from the LLM failed validation!" f"{error_message}"
        )

    else:
        result = chain.invoke({"topic": topic})

        assert result == output
