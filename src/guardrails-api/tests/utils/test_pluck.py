from guardrails_api.utils.pluck import pluck


def test_pluck():
    input = {"a": 1, "b": 2, "c": 3}
    response = pluck(input, ["a", "c"])
    assert response == [1, 3]
