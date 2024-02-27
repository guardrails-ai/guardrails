import sys
from typing import List
from pydantic import BaseModel
import pytest
from lxml import etree as ET

from guardrails.schema.json_schema import JsonSchema

# Set up XML parser
XMLPARSER = ET.XMLParser(encoding="utf-8")


def test_json_schema_from_xml_outermost_list():
    rail_spec = """
<output>
    <list name="temp_name">
        <string name="string_name" />
    </list>
</output>
"""
    try:
        xml = ET.fromstring(rail_spec, parser=XMLPARSER)
        JsonSchema.from_xml(xml)
    except Exception as e:
        pytest.fail(f"JsonSchema.from_xml() raised an exception: {e}")


def test_json_schema_from_pydantic_outermost_list_typing():
    class Foo(BaseModel):
        field: str
    
    # Test 1: typing.List with BaseModel
    try:
        JsonSchema.from_pydantic(model=List[Foo])
    except Exception as e:
        pytest.fail(f"JsonSchema.from_pydantic() raised an exception: {e}")


@pytest.mark.skipif(
        sys.version_info.major <= 3 and sys.version_info.minor <= 8,
        reason="requires Python > 3.8"
)
def test_json_schema_from_pydantic_outermost_list():
    class Foo(BaseModel):
        field: str
    
    # Test 1: typing.List with BaseModel
    try:
        JsonSchema.from_pydantic(model=list[Foo])
    except Exception as e:
        pytest.fail(f"JsonSchema.from_pydantic() raised an exception: {e}")
