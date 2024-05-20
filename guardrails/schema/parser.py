from typing import Any, Dict, List, Optional, Union

from guardrails.utils.safe_get import safe_get


### Reading and Writing Payloads by JSON Path ###
def get_value_from_path(
    object: Optional[Union[str, List[Any], Dict[Any, Any]]], property_path: str
) -> Any:
    if object is None:
        return None

    if isinstance(object, str) and property_path == "$.string":
        return object

    path_elems = property_path.split(".")
    path_elems.pop(0)

    value = object
    for elem in path_elems:
        obj_value = safe_get(value, elem)
        if not obj_value and elem.isnumeric():
            # value was empty but the key may be an array index
            value = safe_get(value, int(elem))
        else:
            value = obj_value

    return value


def fill_list(desired_length: int, array: list):
    while len(array) < (desired_length + 1):
        array.append(None)

    return array


def write_value_to_path(
    write_object: Optional[Union[str, List[Any], Dict[Any, Any]]],
    property_path: str,
    value: Any,
) -> Any:
    if property_path == "$" or not len(property_path):
        return value

    path_elems = property_path.split(".")
    if path_elems[0] == "$":
        path_elems.pop(0)

    this_key = path_elems.pop(0)
    next_key: str = safe_get(path_elems, 0, "")

    remaining_path = ".".join(path_elems)

    next_key_is_index = next_key.isnumeric()
    key = int(this_key) if isinstance(write_object, list) else this_key
    default_value = [] if next_key_is_index else {}
    value_for_key = safe_get(write_object, key, default_value)

    if isinstance(write_object, list) and key >= len(write_object):
        write_object = fill_list(key, write_object)

    write_object[key] = write_value_to_path(value_for_key, remaining_path, value)

    return write_object


### Reading JSON Sub-Schemas ###
# TODO
def traverse_json_schema():
    pass
