# noqa:W291
import pytest

from guardrails import Guard
from guardrails.datatypes import DataType
from guardrails.schema import StringSchema
from guardrails.validators import SimilarToList

from .mock_embeddings import MOCK_EMBEDDINGS


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
    validators = data_type.format_attr.validators
    validator: SimilarToList = validators[0]

    assert isinstance(validator._standard_deviations, int)
    assert isinstance(validator._threshold, float)

    # 1. Test for integer values
    # 1.1 Test for values within the standard deviation
    val = 3
    output = guard.parse(
        llm_output=val,
        metadata={"prev_values": int_prev_values},
    )
    assert int(output) == val

    # 1.2 Test not passing prev_values
    # Should raise ValueError
    with pytest.raises(ValueError):
        val = 3
        output = guard.parse(
            llm_output=val,
        )

    # 1.3 Test passing str prev values for int val
    # Should raise ValueError
    with pytest.raises(ValueError):
        val = 3
        output = guard.parse(
            llm_output=val,
            metadata={"prev_values": [str(i) for i in int_prev_values]},
        )

    # 1.4 Test for values outside the standard deviation
    val = 300
    output = guard.parse(
        llm_output=val,
        metadata={"prev_values": int_prev_values},
    )
    assert output is None

    # 2. Test for string values
    # 2.1 Test for values within the standard deviation
    val = "cisco"
    output = guard.parse(
        llm_output=val,
        metadata={"prev_values": str_prev_values, "embed_function": embed_function},
    )
    assert output == val

    # 2.2 Test not passing prev_values
    # Should raise ValueError
    with pytest.raises(ValueError):
        val = "cisco"
        output = guard.parse(
            llm_output=val,
            metadata={"embed_function": embed_function},
        )

    # 2.3 Test passing int prev values for str val
    # Should raise ValueError
    with pytest.raises(ValueError):
        val = "cisco"
        output = guard.parse(
            llm_output=val,
            metadata={"prev_values": int_prev_values, "embed_function": embed_function},
        )

    # 2.4 Test not pasisng embed_function
    # Should raise ValueError
    with pytest.raises(ValueError):
        val = "cisco"
        output = guard.parse(
            llm_output=val,
            metadata={"prev_values": str_prev_values},
        )

    # 2.5 Test for values outside the standard deviation
    val = "taj mahal"
    output = guard.parse(
        llm_output=val,
        metadata={"prev_values": str_prev_values, "embed_function": embed_function},
    )
    assert output is None
