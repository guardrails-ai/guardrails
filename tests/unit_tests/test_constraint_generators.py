from guardrails.constraint_generator import BalancedBracesGenerator


def test_enforce_balanced_braces():
    constraint = BalancedBracesGenerator(max_depth=1)
    # Text now: ""
    assert constraint.get_valid_tokens() == {"{", "}"}
    # "{"
    constraint.update_valid_tokens("{")
    assert constraint.get_valid_tokens() == {"}"}
    constraint.update_valid_tokens("}")
    # "{}"
    assert constraint.get_valid_tokens() == {"{", "}"}
    constraint.update_valid_tokens("}")
    # "{}}" - No way we can get back to normal now. Empty set of valid tokens.
    assert constraint.get_valid_tokens() == set()
