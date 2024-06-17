import pytest

from guardrails.utils.validator_utils import parse_rail_validator
from guardrails.validator_base import (
    Validator,
    register_validator,
)


@register_validator("test-validator", "all")
class TestValidator(Validator):
    def __init__(self, *args, **kwargs):
        args = args or []
        super().__init__(on_fail=None, args=list(args))
        self.args = list(args)


# TODO: Move to tests/unit_tests/utils/test_validator_utils.py
@pytest.mark.parametrize(
    "input_string, expected",
    [
        ("test-validator: a", ["a"]),
        ("test-validator: a b", ["a", "b"]),
        (
            "test-validator: {list(range(5))} a b",
            [[0, 1, 2, 3, 4], "a", "b"],
        ),
        ("test-validator: {[1, 2, 3]} a b", [[1, 2, 3], "a", "b"]),
        (
            "test-validator: {{'a': 1, 'b': 2}} c d",
            [{"a": 1, "b": 2}, "c", "d"],
        ),
        (
            "test-validator: {1 + 2} {{'a': 1, 'b': 2}} c d",
            [3, {"a": 1, "b": 2}, "c", "d"],
        ),
    ],
)
def test_get_args(input_string, expected):
    validator: TestValidator = parse_rail_validator(input_string)
    print("\n actual: \n", validator.args)
    print("\n expected: \n", expected)
    assert validator.args == expected
