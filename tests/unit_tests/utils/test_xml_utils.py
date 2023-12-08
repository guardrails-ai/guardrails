import pytest

from guardrails.utils.xml_utils import cast_xml_to_string


@pytest.mark.parametrize(
    "xml_input, expected_output",
    [
        # str
        ("hello", "hello"),
        # bytes
        (b"hello", "hello"),
        # bytearray
        (bytearray(b"hello"), "hello"),
        # memoryview
        (memoryview(b"hello"), "hello"),
    ],
)
def test_xml_element_cast(xml_input, expected_output):
    cast_output = cast_xml_to_string(xml_input)
    assert cast_output == expected_output
    assert isinstance(cast_output, str)
