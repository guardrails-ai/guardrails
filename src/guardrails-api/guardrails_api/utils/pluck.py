from typing import List


def pluck(input: dict, keys: List[str]):
    return [input.get(key) for key in keys]
