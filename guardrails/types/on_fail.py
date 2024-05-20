from enum import Enum


class OnFailAction(str, Enum):
    REASK = "reask"
    FIX = "fix"
    FILTER = "filter"
    REFRAIN = "refrain"
    NOOP = "noop"
    EXCEPTION = "exception"
    FIX_REASK = "fix_reask"
