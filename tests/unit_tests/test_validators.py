import pytest

from guardrails.validators import (
    BugFreeSQL,
    EventDetail,
    Filter,
    Refrain,
    SimilarToDocument,
    check_refrain_in_dict,
    filter_in_dict,
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
    schema = {"key": summary}
    assert val.validate("key", summary, schema) == schema


class TestBugFreeSQLValidator:
    def test_bug_free_sql(self):
        # TODO Make this robust by computing the abs path of the sql file
        # relative to this file
        val = BugFreeSQL(
            schema_file="./tests/unit_tests/test_assets/valid_schema.sql",
            conn="sqlite://",
        )
        bad_query = "select name, fro employees"
        with pytest.raises(EventDetail) as context:
            val.validate("sql-query", bad_query, {})
        assert context.type is EventDetail
        assert context.value.error_message != ""

        good_query = "select name from employees;"
        val.validate("sql-query", good_query, {})

    def test_long_sql_schema_no_exception(self):
        val = BugFreeSQL(
            schema_file="./tests/unit_tests/test_assets/spider.sql",
            conn="sqlite://",
        )
        assert val is not None

    def test_bug_free_sql_simple(self):
        val = BugFreeSQL()
        bad_query = "select name, fro employees"
        with pytest.raises(EventDetail) as context:
            val.validate("sql-query", bad_query, {})
        assert context.type is EventDetail
        assert context.value.error_message != ""

        good_query = "select name from employees;"
        val.validate("sql-query", good_query, {})
