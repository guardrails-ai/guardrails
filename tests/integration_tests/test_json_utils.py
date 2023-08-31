import pytest

from guardrails.utils.json_utils import (
    ChoicePlaceholder,
    DictPlaceholder,
    ListPlaceholder,
    ValuePlaceholder,
)


@pytest.mark.parametrize(
    "optional,type_string,value,coerce_types,expected_value",
    [
        (False, "integer", None, True, ValuePlaceholder.VerificationFailed),
        (False, "integer", None, False, ValuePlaceholder.VerificationFailed),
    ],
)
def test_value_placeholder_verify(
    optional, type_string, value, coerce_types, expected_value
):
    ph = ValuePlaceholder(optional, type_string)

    verified_type = ph.verify(value, False, coerce_types)

    assert verified_type == expected_value


@pytest.mark.parametrize(
    "optional,children,value,coerce_types,expected_value",
    [
        (True, {}, None, True, True),
        (False, {}, None, False, False),
        (
            False,
            {"child": ValuePlaceholder(False, "integer")},
            {"child": None},
            False,
            False,
        ),
    ],
)
def test_dict_placeholder_verify(
    optional, children, value, coerce_types, expected_value
):
    ph = DictPlaceholder(optional, children)

    verified_type = ph.verify(value, False, coerce_types)

    assert verified_type == expected_value


@pytest.mark.parametrize(
    "optional,children,value,coerce_types,expected_value",
    [
        (True, None, None, True, True),
        (False, None, None, False, False),
        (False, ValuePlaceholder(False, "integer"), [None], False, False),
    ],
)
def test_list_placeholder_verify(
    optional, children, value, coerce_types, expected_value
):
    ph = ListPlaceholder(optional, children)

    verified_type = ph.verify(value, False, coerce_types)

    assert verified_type == expected_value


@pytest.mark.parametrize(
    "optional,cases,value,coerce_types,expected_value",
    [
        (True, {}, None, True, True),
        (False, {}, None, False, False),
        (False, {}, {}, False, False),
        (False, {}, {"discriminator": None}, False, False),
    ],
)
def test_choice_placeholder_verify(
    optional, cases, value, coerce_types, expected_value
):
    ph = ChoicePlaceholder(optional, "discriminator", cases)

    verified_type = ph.verify(value, False, coerce_types)

    assert verified_type == expected_value


@pytest.mark.parametrize(
    "optional,cases,value,coerce_types,expected_value",
    [
        (
            False,
            {"discriminator": "abc", "abc": ValuePlaceholder(False, "integer")},
            {"discriminator": "abc", "abc": None},
            False,
            "Choice cases must be objects",
        ),
        (
            False,
            {
                "discriminator": "abc",
                "abc": DictPlaceholder(
                    False, {"discriminator": ValuePlaceholder(False, "integer")}
                ),
            },
            {"discriminator": "abc", "abc": {"discriminator": 1}},
            False,
            "Choice cases must be objects",
        ),
    ],
)
def test_choice_placeholder_verify_raises(
    optional, cases, value, coerce_types, expected_value
):
    with pytest.raises(ValueError) as error:
        ph = ChoicePlaceholder(optional, "discriminator", cases)

        return_value = ph.verify(value, False, coerce_types)

        print("return_value: ", return_value)

        import traceback

        traceback.print_exception(error)
        assert str(error) == expected_value
