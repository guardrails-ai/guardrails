from typing import List

from pydantic import BaseModel

LIST_PROMPT = """Create a list of items that may be found in a grocery store."""


LIST_OUTPUT = """[{"name": "apple", "price": 1.0}, {"name": "banana", "price": 0.5}, {"name": "orange", "price": 1.5}]"""  # noqa: E501


class Item(BaseModel):
    name: str
    price: float


PYDANTIC_RAIL_WITH_LIST = List[Item]


message = (
    '<message role="user">Create a list of items that may be found in a grocery store.</message>'
)
RAIL_SPEC_WITH_LIST = f"""
<rail version="0.1">
  <output type="list">
    <object>
        <string name="name" />
        <float name="price" />
    </object>
  </output>
  <messages>
  {message}
  </messages>
</rail>
"""
