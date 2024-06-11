from guardrails.constraint_generator import BalancedBracesGenerator


def test_enforce_balanced_braces():
    constraint = BalancedBracesGenerator(max_depth=2)
    # Text now: ""
    # We can't close an unopened paren, so the only valid next token is '{'.
    assert constraint.get_valid_tokens() == {"{"}
    constraint.update_valid_tokens("{")
    # "{"
    assert constraint.get_valid_tokens() == {"{", "}"}  # Could open or close.
    constraint.update_valid_tokens("}")
    # "{}"
    constraint.update_valid_tokens("}")
    # "{}}" - No way we can get back to normal now. Empty set of valid tokens.
    assert constraint.get_valid_tokens() == set()
