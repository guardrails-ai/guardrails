from typing import Any


def to_int(v: Any) -> int:
    try:
        int_value = int(v)
        return int_value
    except Exception:
        return None


def to_float(v: Any) -> float:
    try:
        float_value = float(v)
        return float_value
    except Exception:
        return None


def to_string(v: Any) -> str:
    try:
        str_value = str(v)
        return str_value
    except Exception:
        return None
