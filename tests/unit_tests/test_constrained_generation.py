import pytest

from guardrails.constrained_generation import (
    BalancedBracesGenerator,
    NumberConstraintGenerator,
    JSONValueConstraint,
    KeywordConstraintGenerator,
    QuotedStringConstraintGenerator,
    UnionConstraintGenerator,
)


def test_enforce_balanced_braces():
    constraint = BalancedBracesGenerator(max_depth=2)
    # Text now: ""
    # We can't close an unopened paren, so the only valid next token is '{'.
    assert constraint.get_valid_tokens() == {"{"}
    constraint.update_valid_tokens("{")
    # "{"
    assert constraint.get_valid_tokens() == {"{", "}"}  # Could open or close.
    assert not constraint.is_complete()
    constraint.update_valid_tokens("}")
    assert constraint.is_complete()
    # "{}"
    constraint.update_valid_tokens("}")
    # "{}}" - No way we can get back to normal now. Empty set of valid tokens.
    assert constraint.get_valid_tokens() == set()
    assert not constraint.is_complete()


def test_integer_constraint():
    # Make sure all our digits are valid.
    c = NumberConstraintGenerator(is_integer=True)
    for i in range(0, 10):
        assert str(i) in c.get_valid_tokens()
    # An integer can start with '-'
    assert "-" in c.get_valid_tokens()
    # But if we add a digit then it can't. -12 is valid. 1-2 is not.
    c.update_valid_tokens("1")
    assert "-" not in c.get_valid_tokens()


@pytest.mark.parametrize(
    "number,is_integer,is_valid",
    [
        ("1234567890", True, True),
        ("-12", True, True),
        ("1234", False, True),
        ("1-2", True, False),
        ("1.234", False, True),
        ("0.1234", True, False),
        ("-1234.567890", False, True),
    ],
)
def test_float_constraint(number: str, is_integer: bool, is_valid: bool):
    all_valid = True
    c = NumberConstraintGenerator(is_integer=is_integer)
    for digit in number:
        if digit not in c.get_valid_tokens():
            all_valid = False
        c.update_valid_tokens(digit)
    assert is_valid == all_valid
    assert is_valid == c.is_complete()


def test_keyword_constraint():
    c = KeywordConstraintGenerator("Outstanding", token_length_cap=3)
    assert c.get_valid_tokens() == {"O", "Ou", "Out"}
    c.update_valid_tokens("Out")
    assert c.get_valid_tokens() == {"s", "st", "sta"}


def test_true_or_false_keyword_constraint():
    false_keyword = KeywordConstraintGenerator("False", token_length_cap=1)
    true_keyword = KeywordConstraintGenerator("True", token_length_cap=1)
    c = UnionConstraintGenerator(false_keyword, true_keyword)
    # We can have either "True" or "False" until we parse a 'T' or 'F'.
    assert c.get_valid_tokens() == {"T", "F"}
    c.update_valid_tokens("T")
    assert c.get_valid_tokens() == {"r"}
    c.update_valid_tokens("rue")
    assert c.get_valid_tokens() == set()
    assert c.is_complete()


def test_quoted_string_constraint():
    c = QuotedStringConstraintGenerator()
    assert c.get_valid_tokens() == {'"'}
    c.update_valid_tokens('"')
    assert c.get_valid_tokens() is None  # No constraints
    c.update_valid_tokens("simple_quote test with space")
    assert not c.is_complete()
    assert c.get_valid_tokens() is None  # No constraints
    c.update_valid_tokens('"')
    assert c.is_complete()


def test_quoted_string_with_escapes():
    c = QuotedStringConstraintGenerator()
    c.update_valid_tokens('"This starts with a double quote')
    assert not c.is_complete()
    c.update_valid_tokens(', and has \\"one escaped quote')
    assert not c.is_complete()
    c.update_valid_tokens('and ENDS with an escaped double quote!\\"')
    assert not c.is_complete()
    c.update_valid_tokens(' and is finally complete."')
    assert c.is_complete()


def test_json_value_constraint():
    c = JSONValueConstraint()
    assert c.get_valid_tokens() == {'"'}
    assert not c.is_complete()
    c.update_valid_tokens('"foo"')
    assert c.get_valid_tokens() == {":"}
    assert not c.is_complete()
    c.update_valid_tokens(":")
    assert c.get_valid_tokens() == set('"tfn-1234567890')
    assert not c.is_complete()
    c.update_valid_tokens('"')  # Starting a quoted string.
    assert c.get_valid_tokens() is None
    assert not c.is_complete()
    c.update_valid_tokens('bar"')
    assert c.is_complete()

    c = JSONValueConstraint()
    c.update_valid_tokens('"joke123":"Why do mirrors look like eyeballs up close?"')
    assert c.get_valid_tokens() == set()
    assert c.is_complete()
