from typing import Dict, Optional
from types import SimpleNamespace


from lxml import etree as ET

from guardrails.datatypes import DataType


class Response:
    """Response schema class that holds a _schema attribute.
    
    Calling response['a'] = 'b' should set response._schema['a'] = 'b'.
    I should be able to do response.a = 'b' and have it set response._schema.a = 'b'.
    """
    def __init__(
        self,
        parsed_aiml: Optional[ET._Element] = None,
        schema: Optional[Dict[str, DataType]] = None,
    ) -> None:

        if schema is None:
            schema = {}

        self._schema = SimpleNamespace(**schema)
        self.parsed_aiml = parsed_aiml

    def __getitem__(self, key: str) -> DataType:
        return getattr(self._schema, key)
    
    def __setitem__(self, key: str, value: DataType) -> None:
        setattr(self._schema, key, value)

    def __getattr__(self, key: str) -> DataType:
        return getattr(self._schema, key)

    def items(self) -> Dict[str, DataType]:
        return vars(self._schema).items()