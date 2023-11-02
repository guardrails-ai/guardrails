# noqa:W291
import os
from typing import Any, Dict

import openai
import pytest
from pydantic import BaseModel, Field

from guardrails import Guard
from guardrails.datatypes import DataType
from guardrails.schema import StringSchema
from guardrails.utils.reask_utils import FieldReAsk
from guardrails.validator_base import (
    FailResult,
    Filter,
    PassResult,
    Refrain,
    ValidationResult,
    check_refrain_in_dict,
    filter_in_dict,
    register_validator,
)
from guardrails.validators import (
    BugFreeSQL,
    DetectSecrets,
    ExtractedSummarySentencesMatch,
    ExtractiveSummary,
    ProvenanceV1,
    SimilarToDocument,
    SimilarToList,
    SqlColumnPresence,
    TwoWords,
    ValidLength,
)

from .mock_embeddings import MOCK_EMBEDDINGS, mock_create_embedding
from .mock_provenance_v1 import mock_chat_completion, mock_chromadb_query_function
from .mock_secrets import (
    EXPECTED_SECRETS_CODE_SNIPPET,
    NO_SECRETS_CODE_SNIPPET,
    SECRETS_CODE_SNIPPET,
)


@pytest.mark.parametrize(
    "input_dict, expected",
    [
        ({"a": 1, "b": Refrain()}, True),
        ({"a": 1, "b": {"c": 2, "d": Refrain()}}, True),
        ({"a": [1, 2, Refrain()], "b": 4}, True),
        ({"a": [1, 2, {"c": Refrain()}]}, True),
        ({"a": [1, 2, [3, 4, Refrain()]]}, True),
        ({"a": 1}, False),
    ],
)
def test_check_refrain(input_dict, expected):
    assert check_refrain_in_dict(input_dict) == expected


@pytest.mark.parametrize(
    "input_dict, expected_dict",
    [
        ({"a": 1, "b": Filter(), "c": 3}, {"a": 1, "c": 3}),
        (
            {"a": 1, "b": {"c": 2, "d": Filter()}, "e": 4},
            {"a": 1, "b": {"c": 2}, "e": 4},
        ),
        ({"a": [1, 2, Filter()], "b": 4}, {"a": [1, 2], "b": 4}),
        ({"a": [1, 2, {"c": Filter(), "d": 3}]}, {"a": [1, 2, {"d": 3}]}),
        ({"a": [1, 2, [3, 4, Filter()]]}, {"a": [1, 2, [3, 4]]}),
        ({"a": 1}, {"a": 1}),
    ],
)
def test_filter_in_dict(input_dict, expected_dict):
    assert filter_in_dict(input_dict) == expected_dict


# TODO: Implement testing with models on CI
@pytest.mark.skip(
    reason="This test requires the text-embedding-ada-002 embedding model."
    " Testing with models needs to be implemented."
)
def test_similar_to_document_validator():
    import os

    datapath = os.path.abspath(os.path.dirname(__file__) + "/../data/article1.txt")
    val = SimilarToDocument(
        document=open(datapath, "r").read(),
        model="text-embedding-ada-002",
        threshold=0.85,
    )
    summary = "All legislative powers are held by a Congress"
    " consisting of two chambers, the Senate and the House of Representatives."
    assert isinstance(val.validate(summary, {}), PassResult)


class TestBugFreeSQLValidator:
    def test_bug_free_sql(self):
        # TODO Make this robust by computing the abs path of the sql file
        # relative to this file
        val = BugFreeSQL(
            schema_file="./tests/unit_tests/test_assets/valid_schema.sql",
            conn="sqlite://",
        )
        bad_query = "select name, fro employees"
        result = val.validate(bad_query, {})
        assert isinstance(result, FailResult)
        assert result.error_message != ""

        good_query = "select name from employees;"
        assert isinstance(val.validate(good_query, {}), PassResult)

    def test_long_sql_schema_no_exception(self):
        val = BugFreeSQL(
            schema_file="./tests/unit_tests/test_assets/spider.sql",
            conn="sqlite://",
        )
        assert val is not None

    def test_bug_free_sql_simple(self):
        val = BugFreeSQL()
        bad_query = "select name, fro employees"

        result = val.validate(bad_query, {})
        assert isinstance(result, FailResult)
        assert result.error_message != ""

        good_query = "select name from employees;"
        assert isinstance(val.validate(good_query, {}), PassResult)

    def test_sql_column_presense(self):
        sql = "select name, age from employees;"
        columns = ["name", "address"]
        val = SqlColumnPresence(cols=columns)

        result = val.validate(sql, {})
        assert isinstance(result, FailResult)
        assert result.error_message in (
            "Columns [age] not in [name, address]",
            "Columns [age] not in [address, name]",
        )


def test_summary_validators(mocker):
    pytest.importorskip("nltk", reason="nltk is not installed")
    pytest.importorskip("thefuzz", reason="thefuzz is not installed")

    mocker.patch("openai.Embedding.create", new=mock_create_embedding)
    mocker.patch("guardrails.embedding.OpenAIEmbedding.output_dim", new=2)

    summary = "It was a nice day. I went to the park. I saw a dog."
    metadata = {
        "filepaths": [
            "./tests/unit_tests/test_assets/article1.txt",
            "./tests/unit_tests/test_assets/article2.txt",
        ]
    }

    val = ExtractedSummarySentencesMatch(threshold=0.1)
    result = val.validate(summary, metadata)
    assert isinstance(result, PassResult)
    assert "citations" in result.metadata
    assert "summary_with_citations" in result.metadata
    assert result.metadata["citations"] == {1: 1, 2: 1, 3: 1}
    assert (
        result.metadata["summary_with_citations"]
        == """It was a nice day. [1] I went to the park. [1] I saw a dog. [1]

[1] ./tests/unit_tests/test_assets/article1.txt
[2] ./tests/unit_tests/test_assets/article2.txt"""
    )

    val = ExtractiveSummary(
        threshold=30,
    )
    result = val.validate(summary, metadata)
    assert isinstance(result, PassResult)
    assert "citations" in result.metadata
    assert "summary_with_citations" in result.metadata
    assert result.metadata["citations"] == {1: 1, 2: 2, 3: 1}
    assert (
        result.metadata["summary_with_citations"]
        == """It was a nice day. [1] I went to the park. [2] I saw a dog. [1]

[1] ./tests/unit_tests/test_assets/article1.txt
[2] ./tests/unit_tests/test_assets/article2.txt"""
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
        a_field: str = Field(..., validators=[(hello_validator, "fix")])

    guard = Guard.from_pydantic(MyModel)
    output = guard.parse(
        '{"a_field": "hello there yo"}',
        num_reasks=0,
    )

    assert output.validated_output == {"a_field": "hullo"}

    # (string, on_fail) tuple fix

    class MyModel(BaseModel):
        a_field: str = Field(
            ..., validators=[("two_words", "reask"), ("mycustomhellovalidator", "fix")]
        )

    guard = Guard.from_pydantic(MyModel)
    output = guard.parse(
        '{"a_field": "hello there yo"}',
        num_reasks=0,
    )

    assert output.validated_output == {"a_field": "hullo"}

    # (Validator, on_fail) tuple fix

    class MyModel(BaseModel):
        a_field: str = Field(..., validators=[(TwoWords(), "fix")])

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
        a_field: str = Field(..., validators=[(hello_validator, "reask")])

    guard = Guard.from_pydantic(MyModel)

    output = guard.parse(
        '{"a_field": "hello there yo"}',
        num_reasks=0,
    )

    assert output.validated_output == {"a_field": "hullo"}
    assert guard.guard_state.all_histories[0].history[0].reasks[0] == hullo_reask

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
        a_field: str = Field(..., validators=[("two-words", "reask")])

    guard = Guard.from_pydantic(MyModel)

    output = guard.parse(
        '{"a_field": "hello there yo"}',
        num_reasks=0,
    )

    assert output.validated_output == {"a_field": "hello there"}
    assert guard.guard_state.all_histories[0].history[0].reasks[0] == hello_reask

    # (Validator, on_fail) tuple reask

    class MyModel(BaseModel):
        a_field: str = Field(..., validators=[(TwoWords(), "reask")])

    guard = Guard.from_pydantic(MyModel)

    output = guard.parse(
        '{"a_field": "hello there yo"}',
        num_reasks=0,
    )

    assert output.validated_output == {"a_field": "hello there"}
    assert guard.guard_state.all_histories[0].history[0].reasks[0] == hello_reask

    # Fail on string

    class MyModel(BaseModel):
        a_field: str = Field(..., validators=["two-words"])

    with pytest.raises(ValueError):
        Guard.from_pydantic(MyModel)


def test_custom_func_validator():
    rail_str = """
    <rail version="0.1">
    <output>
        <string name="greeting"
                format="mycustomhellovalidator"
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

    guard_history = guard.guard_state.all_histories[0].history
    assert len(guard_history) == 1
    validator_log = (
        guard_history[0].field_validation_logs.children["greeting"].validator_logs[0]
    )
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


def test_provenance_v1(mocker):
    """Test initialisation of ProvenanceV1."""

    mocker.patch("openai.ChatCompletion.create", new=mock_chat_completion)
    API_KEY = "<YOUR_KEY>"
    LLM_RESPONSE = "This is a sentence."

    # Initialise Guard from string
    string_guard = Guard.from_string(
        validators=[
            ProvenanceV1(
                validation_method="full",
                llm_callable="gpt-3.5-turbo",
                top_k=3,
                max_tokens=100,
                on_fail="fix",
            )
        ],
        description="testmeout",
    )

    output_schema: StringSchema = string_guard.rail.output_schema
    data_type: DataType = output_schema.root_datatype
    validators = data_type.format_attr.validators
    prov_validator: ProvenanceV1 = validators[0]

    # Check types remain intact
    assert isinstance(prov_validator._validation_method, str)
    assert isinstance(prov_validator._top_k, int)
    assert isinstance(prov_validator._max_tokens, int)

    # Test guard.parse() with 3 different ways of setting the OpenAI API key API key
    # 1. Setting the API key directly
    openai.api_key = API_KEY

    output = string_guard.parse(
        llm_output=LLM_RESPONSE,
        metadata={"query_function": mock_chromadb_query_function},
    )
    assert output.validated_output == LLM_RESPONSE

    # 2. Setting the environment variable
    os.environ["OPENAI_API_KEY"] = API_KEY
    output = string_guard.parse(
        llm_output=LLM_RESPONSE,
        metadata={"query_function": mock_chromadb_query_function},
    )
    assert output.validated_output == LLM_RESPONSE

    # 3. Passing the API key as an argument
    output = string_guard.parse(
        llm_output=LLM_RESPONSE,
        metadata={"query_function": mock_chromadb_query_function},
        api_key=API_KEY,
        api_base="https://api.openai.com",
    )
    assert output.validated_output == LLM_RESPONSE


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


def test_similar_to_list():
    # Mock embedding function
    def embed_function(text: str):
        """Mock embedding function."""
        return MOCK_EMBEDDINGS[text]

    # Initialise validator
    validator = SimilarToList()

    # Test get_semantic_similarity method
    similarity = validator.get_semantic_similarity(
        "broadcom", "broadcom", embed_function
    )
    # Assert that similarity is very close to 0
    assert similarity == pytest.approx(0.0, abs=1e-2)


def test_detect_secrets():
    """Test the DetectSecrets validator.

    1. Test with dummy code snippet with secrets
    2. Test with dummy code snippet without secrets

    No mock functions are used in this test, as we are testing the actual
    functionality of the detect_secrets package, which is used by the
    DetectSecrets validator.
    """
    # Initialise validator
    validator = DetectSecrets()

    # ----------------------------
    # 1. Test get_unique_secrets and get_modified_value
    # with dummy code snippet with secrets
    unique_secrets, lines = validator.get_unique_secrets(SECRETS_CODE_SNIPPET)

    # Check types of unique_secrets and lines
    assert isinstance(unique_secrets, dict)
    assert isinstance(lines, list)

    # Check if unique_secrets contains exactly 2 secrets
    assert len(unique_secrets.keys()) == 2

    # Check if lines contains exactly 7 lines
    assert len(lines) == 7

    # Check if temp.txt does not exist in current directory
    assert not os.path.exists(validator.temp_file_name)

    mod_value = validator.get_modified_value(unique_secrets, lines)
    assert mod_value != SECRETS_CODE_SNIPPET
    assert mod_value == EXPECTED_SECRETS_CODE_SNIPPET

    # ----------------------------
    # 2. Test get_unique_secrets and get_modified_value
    # with dummy code snippet without secrets
    unique_secrets, lines = validator.get_unique_secrets(NO_SECRETS_CODE_SNIPPET)

    # Check types of unique_secrets and lines
    assert isinstance(unique_secrets, dict)
    assert isinstance(lines, list)

    # Check if unique_secrets is empty
    assert len(unique_secrets.keys()) == 0

    # Check if lines contains exactly 10 lines
    assert len(lines) == 10

    # Check if temp.txt does not exist in current directory
    assert not os.path.exists(validator.temp_file_name)

    mod_value = validator.get_modified_value(unique_secrets, lines)

    # Check if mod_value is same as code_snippet,
    # as there are no secrets in code_snippet
    assert mod_value == NO_SECRETS_CODE_SNIPPET
