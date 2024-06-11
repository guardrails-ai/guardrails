from typing import Optional, Union


# TODO: Remove after DataTypes and ValidatorsAttr is removed
def cast_xml_to_string(xml_value: Union[memoryview, bytes, bytearray, str]) -> str:
    """Cast XML value to a string.

    Args:
        xml_value (Union[memoryview, bytes, bytearray, str]): The XML value to cast.

    Returns:
        str: The XML value as a string.
    """
    return xml_to_string(xml_value)


def xml_to_string(
    xml: Optional[Union[memoryview, bytes, bytearray, str]],
) -> Optional[str]:
    """Convert XML value to a string.

    Args:
        xml_value (Union[memoryview, bytes, bytearray, str]): The XML value to cast.

    Returns:
        str: The XML value as a string.
    """
    string = xml
    if isinstance(xml, memoryview):
        string = xml.tobytes().decode()
    elif isinstance(xml, (bytes, bytearray)):
        string = xml.decode()
    return string
