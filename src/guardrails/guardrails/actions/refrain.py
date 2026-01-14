from typing import Any, Dict, List, Union
from guardrails.classes.output_type import OutputTypes
from guardrails.logger import logger


class Refrain:
    pass


def check_for_refrain(value: Union[List, Dict]) -> bool:
    if isinstance(value, Refrain):
        return True
    elif isinstance(value, list):
        for item in value:
            if check_for_refrain(item):
                return True
    elif isinstance(value, dict):
        for key, child in value.items():
            if check_for_refrain(child):
                return True

    return False


# Could be a generic instead of Any
def apply_refrain(value: Any, output_type: OutputTypes) -> Any:
    """Recursively check for any values that are instances of Refrain.

    If found, return an empty value of the appropriate type.
    """
    refrain_value = {}
    if output_type == OutputTypes.STRING:
        refrain_value = ""
    elif output_type == OutputTypes.LIST:
        refrain_value = []

    if check_for_refrain(value):
        # If the data contains a `Refain` value, we return an empty
        # value.
        logger.debug("Refrain detected.")
        value = refrain_value

    return value
