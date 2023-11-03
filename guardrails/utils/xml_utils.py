from typing import Union


def cast_xml_to_string(xml_value: Union[memoryview, bytes, bytearray, str]) -> str:
    """Cast XML value to a string.

    Args:
        xml_value (Union[memoryview, bytes, bytearray, str]): The XML value to cast.

    Returns:
        str: The XML value as a string.
    """
    if isinstance(xml_value, memoryview):
        xml_value = xml_value.tobytes().decode()
    elif isinstance(xml_value, (bytes, bytearray)):
        xml_value = xml_value.decode()
    return xml_value
