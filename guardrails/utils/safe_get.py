from typing import Any, Dict, List, Optional, Tuple, Union


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
    container: Union[str, List[Any], Dict[Any, Any], Tuple],
    key: Any,
    default: Optional[Any] = None,
) -> Any:
    if isinstance(container, dict):
        return container.get(key, default)
    else:
        return safe_get_with_brackets(container, key, default)
