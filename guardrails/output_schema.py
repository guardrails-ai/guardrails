from types import SimpleNamespace
from typing import Dict, Optional

from lxml import etree as ET

from guardrails.datatypes import DataType


class OutputSchema:
    """Output schema class that holds a _schema attribute."""

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
