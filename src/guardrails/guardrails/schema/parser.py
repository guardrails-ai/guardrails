from guardrails_api_client.models.simple_types import SimpleTypes
import jsonref
from typing import Any, Dict, List, Optional, Set, Union, cast

from guardrails.utils.safe_get import safe_get


### Reading and Writing Payloads by JSON Path ###
def get_value_from_path(
    object: Optional[Union[str, List[Any], Dict[Any, Any]]], property_path: str
) -> Any:
    if object is None:
        return None

    # TODO: Remove this when dummy key is removed
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


# FIXME: Better Typing
def write_value_to_path(
    write_object: Union[str, List[Any], Dict[Any, Any]],
    property_path: str,
    value: Any,
) -> Any:
    if property_path == "$" or not len(property_path) or isinstance(write_object, str):
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

    if isinstance(write_object, list) and int(key) >= len(write_object):
        write_object = fill_list(int(key), write_object)

    write_object[key] = write_value_to_path(value_for_key, remaining_path, value)  # type: ignore

    return write_object


### Reading and Manipulating JSON Schemas ###
def _get_all_paths(
    json_schema: Dict[str, Any],
    *,
    paths: Optional[Set[str]] = None,
    json_path: str = "$",
) -> Set[str]:
    if not paths:
        paths = set()
    # Append the parent path for this iteration
    paths.add(json_path)

    # Object Schema
    schema_properties: Dict[str, Any] = json_schema.get("properties", {})
    for k, v in schema_properties.items():
        child_path = f"{json_path}.{k}"
        _get_all_paths(v, paths=paths, json_path=child_path)

    ## Object Schema allows anonymous properties
    additional_properties: Dict[str, Any] = json_schema.get(
        "additionalProperties", False
    )
    schema_type = json_schema.get("type")
    # NOTE: Technically we should check for schema compositions
    #   that would yield an object as well,
    #   but the case below is a known fault of Pydantic.
    if additional_properties or (
        not json_schema.get("properties") and schema_type == SimpleTypes.OBJECT
    ):
        wildcard_path = f"{json_path}.*"
        paths.add(wildcard_path)

    # Array Schema
    schema_items = json_schema.get("items", {})
    if schema_items:
        _get_all_paths(schema_items, paths=paths, json_path=json_path)

    # Conditional SubSchema
    if_block: Dict[str, Any] = json_schema.get("if", {})
    if if_block:
        _get_all_paths(if_block, paths=paths, json_path=json_path)

    then_block: Dict[str, Any] = json_schema.get("then", {})
    if then_block:
        _get_all_paths(then_block, paths=paths, json_path=json_path)

    else_block: Dict[str, Any] = json_schema.get("else", {})
    if else_block:
        _get_all_paths(else_block, paths=paths, json_path=json_path)

    # Schema Composition
    oneOf: List[Dict[str, Any]] = json_schema.get("oneOf", [])
    for sub_schema in oneOf:
        _get_all_paths(sub_schema, paths=paths, json_path=json_path)

    anyOf: List[Dict[str, Any]] = json_schema.get("anyOf", [])
    for sub_schema in anyOf:
        _get_all_paths(sub_schema, paths=paths, json_path=json_path)

    allOf: List[Dict[str, Any]] = json_schema.get("allOf", [])
    for sub_schema in allOf:
        _get_all_paths(sub_schema, paths=paths, json_path=json_path)

    return paths


def get_all_paths(
    json_schema: Dict[str, Any],
    *,
    paths: Optional[Set[str]] = None,
    json_path: str = "$",
) -> Set[str]:
    """Takes a JSON Schema and returns all possible JSONPaths within that
    schema."""
    dereferenced_schema = cast(Dict[str, Any], jsonref.replace_refs(json_schema))
    return _get_all_paths(dereferenced_schema, paths=paths, json_path=json_path)
