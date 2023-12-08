from typing import Any, Optional


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
