from typing import Any, List


def flatten(l: List[List[Any]]) -> List[Any]:
    return [item for sublist in l for item in sublist]


def merge(l: List[dict]) -> dict:
    return {k: v for d in l for k, v in d.items()}
