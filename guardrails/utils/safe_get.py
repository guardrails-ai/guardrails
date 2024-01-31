from typing import Any, Dict, List, Optional, Union


def safe_get_with_brackets(
    container: Union[str, List[Any], Any], key: Any, default: Optional[Any] = None
) -> Any:
    try:
        value = container[key]
        if not value:
            return default
        return value
    except Exception:
        return default


def safe_get(
    container: Union[str, List[Any], Dict[Any, Any]],
    key: Any,
    default: Optional[Any] = None,
) -> Any:
    if isinstance(container, dict):
        return container.get(key, default)
    else:
        return safe_get_with_brackets(container, key, default)


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
