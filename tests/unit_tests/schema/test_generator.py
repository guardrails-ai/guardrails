from decimal import Decimal
import pytest
from guardrails.schema.generator import gen_num


@pytest.mark.parametrize(
    "schema,min,max,multiple,is_int",
    [
        ({"type": "integer"}, 0, 100, 1, True),
        ({"type": "integer", "minimum": 5}, 5, 100, 1, True),
        ({"type": "integer", "exclusiveMinimum": 5}, 6, 100, 1, True),
        ({"type": "integer", "maximum": 5}, 0, 5, 1, True),
        ({"type": "integer", "exclusiveMaximum": 5}, 0, 4, 1, True),
        ({"type": "integer", "multipleOf": 5}, 0, 100, 5, True),
        ({"type": "number"}, 0, 100, 1, False),
        ({"type": "number", "minimum": 0.314}, 0.314, 100, 0.001, False),
        ({"type": "number", "exclusiveMinimum": 3.14}, 3.15, 100, 0.01, False),
        ({"type": "number", "maximum": 2.718}, 0, 2.718, 0.001, False),
        ({"type": "number", "exclusiveMaximum": 2.718}, 0, 2.717, 0.001, False),
        ({"type": "number", "multipleOf": 0.1}, 0, 100, 0.1, False),
    ],
)
def test_gen_num(schema, min, max, multiple, is_int: bool):
    result = gen_num(schema)
    assert result >= min
    assert result <= max
    # Modulo is unreliable with decimals
    # Assert the division is an integer instead
    div_result = round(Decimal(result) / Decimal(multiple), 3)
    assert div_result % 1 == 0
    assert isinstance(result, int) is is_int
