import logging
from copy import deepcopy
from types import SimpleNamespace
from typing import Any, Dict, Optional

from lxml import etree as ET

from guardrails.datatypes import DataType
from guardrails.validators import check_refrain_in_dict, filter_in_dict

logger = logging.getLogger(__name__)


class Schema:
    """Schema class that holds a _schema attribute."""

    def __init__(
        self,
        parsed_rail: Optional[ET._Element] = None,
        schema: Optional[Dict[str, DataType]] = None,
    ) -> None:
        if schema is None:
            schema = {}

        self._schema = SimpleNamespace(**schema)
        self.parsed_rail = parsed_rail

    def __getitem__(self, key: str) -> DataType:
        return getattr(self._schema, key)

    def __setitem__(self, key: str, value: DataType) -> None:
        setattr(self._schema, key, value)

    def __getattr__(self, key: str) -> DataType:
        return getattr(self._schema, key)

    def __contains__(self, key: str) -> bool:
        return hasattr(self._schema, key)

    def items(self) -> Dict[str, DataType]:
        return vars(self._schema).items()

    @classmethod
    def from_schema(cls, schema: "Schema") -> "Schema":
        """Create an InputSchema from a Schema."""
        return cls(schema.parsed_rail, schema._schema.__dict__)

    def validate(
        self,
        data: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Validate a dictionary of data against the schema.

        Args:
            data: The data to validate.

        Returns:
            Tuple, where the first element is the validated output, and the
            second element is a list of tuples, where each tuple contains the
            path to the reasked element, and the ReAsk object.
        """
        if data is None:
            return None

        if not isinstance(data, dict):
            raise TypeError(f"Argument `data` must be a dictionary, not {type(data)}.")

        validated_response = deepcopy(data)

        for field, value in validated_response.items():
            if field not in self:
                # This is an extra field that is not in the schema.
                # We remove it from the validated response.
                logger.debug(f"Field {field} not in schema.")
                continue

            validated_response = self[field].validate(
                key=field,
                value=value,
                schema=validated_response,
            )

        if check_refrain_in_dict(validated_response):
            # If the data contains a `Refain` value, we return an empty
            # dictionary.
            logger.debug("Refrain detected.")
            validated_response = {}

        # Remove all keys that have `Filter` values.
        validated_response = filter_in_dict(validated_response)

        return validated_response


class InputSchema(Schema):
    """Input schema class that holds a _schema attribute."""


class OutputSchema(Schema):
    """Output schema class that holds a _schema attribute."""
