from enum import Enum
from typing import Optional, Union
from guardrails.logger import logger


class OnFailAction(str, Enum):
    REASK = "reask"
    FIX = "fix"
    FILTER = "filter"
    REFRAIN = "refrain"
    NOOP = "noop"
    EXCEPTION = "exception"
    FIX_REASK = "fix_reask"
    CUSTOM = "custom"

    @staticmethod
    def get(key: Optional[Union[str, "OnFailAction"]], default=None):
        try:
            if not key:
                return default
            if isinstance(key, OnFailAction):
                return key
            return OnFailAction[key]

        except Exception as e:
            logger.debug("Failed to get OnFailAction for key ", key)
            logger.debug(e)
            return default
