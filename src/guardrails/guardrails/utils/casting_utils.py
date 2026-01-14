from typing import Any, Optional
import warnings


def to_int(v: Any) -> Optional[int]:
    try:
        int_value = int(v)
        return int_value
    except Exception:
        return None


def to_float(v: Any) -> Optional[float]:
    try:
        float_value = float(v)
        return float_value
    except Exception:
        return None


def to_string(v: Any) -> Optional[str]:
    try:
        str_value = str(v)
        return str_value
    except Exception:
        return None


def to_bool(value: str) -> Optional[bool]:
    if value.lower() == "true":
        return True
    if value.lower() == "false":
        return False
    warnings.warn(f"Could not cast {value} to bool. Returning None.")
    return None
