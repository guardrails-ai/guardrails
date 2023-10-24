from typing import Any, Dict, List, Optional, Union


def safe_get_with_brackets(
    container: Union[List[Any], Any], key: Any, default: Optional[Any] = None
) -> Any:
    try:
        value = container[key]
        if not value:
            return default
        return value
    except Exception:
        return default


def safe_get(
    container: Union[List[Any], Dict[Any, Any]], key: Any, default: Optional[Any] = None
) -> Any:
    if isinstance(container, dict):
        return container.get(key, default)
    else:
        return safe_get_with_brackets(container, key, default)
