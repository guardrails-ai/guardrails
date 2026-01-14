from guardrails_api.utils.remove_nones import remove_nones


def test_remove_nones():
    dictionary = {
        "complete_dictionary": {"a": 1, "b": 2},
        "partial_dictionary": {"a": 1, "b": None},
        "empty_dictionary": {"a": None, "b": None},
        "complete_list": [1, 2],
        "partial_list": [1, None],
        "empty_list": [None, None],
        "complete_primitive": 1,
        "empty_primitive": None,
    }

    filtered = remove_nones(dictionary)

    assert filtered == {
        "complete_dictionary": {"a": 1, "b": 2},
        "partial_dictionary": {"a": 1},
        "empty_dictionary": {},
        "complete_list": [1, 2],
        "partial_list": [1],
        "empty_list": [],
        "complete_primitive": 1,
    }
