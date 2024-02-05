import pytest

from guardrails.validators import EndsWith, FailResult, PassResult


@pytest.mark.parametrize(
    "input, end, outcome, fix_value",
    [
        ("Test string", "g", "pass", None),
        ("Test string", "string", "pass", None),
        (["Item 1", "Item 2"], ["Item 2"], "pass", None),
        (["Item 1", "Item 2", "Item 3"], ["Item 2", "Item 3"], "pass", None),
        ("Test string", "Test", "fail", "Test stringTest"),
        (["Item 1", "Item 2"], ["Item 1"], "fail", ["Item 1", "Item 2", "Item 1"]),
        (["Item 1", "Item 2"], "Item 2", "pass", None),
        (["Item 1", "Item 2"], "Item 3", "fail", ["Item 1", "Item 2", "Item 3"]),
    ],
)
def test_ends_with_validator(input, end, outcome, fix_value):
    """Test that the validator returns the expected outcome and fix_value."""
    validator = EndsWith(end=end, on_fail="fix")
    result: PassResult = validator.validate(input, {})

    # Check that the result matches the expected outcome and fix_value
    if outcome == "fail":
        assert isinstance(result, FailResult)
        assert result.fix_value == fix_value
    else:
        assert isinstance(result, PassResult)
