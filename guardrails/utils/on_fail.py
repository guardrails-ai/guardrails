from typing import Literal, get_args

ON_FAIL_TYPES = Literal[
    "exception", "fix", "fix_reask", "reask", "filter", "refrain", "noop", "custom"
]


def on_fail(fix_type: ON_FAIL_TYPES = "noop"):
    options = get_args(ON_FAIL_TYPES)
    assert fix_type in options, f"'{fix_type}' is not in {options}"
    return {"on_fail": fix_type}
