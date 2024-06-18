from pydantic import BaseModel

from guardrails.schema.pydantic_schema import pydantic_model_to_schema

from guardrails.utils.tools_utils import schema_to_tool


class Delivery(BaseModel):
    name: str
    description: str
    number_of_items: int
    address: str
    price: float

def test_pydantic_model_to_schema():
    schema = pydantic_model_to_schema(Delivery)
    tool = schema_to_tool(schema)
    assert tool == {
        "type": "function",
        "function": {
            "name": "gd_response_tool",
            "description": "A tool for generating responses to guardrails. It must be called last in every response.",
            "parameters": {
                "type": "object",
                "properties": {
                    "name": {
                        "type": "string",
                        "description": "",
                    },
                    "description": {
                        "type": "string",
                        "description": "",
                    },
                    "number_of_items": {
                        "type": "integer",
                        "description": "",
                    },
                    "address": {
                        "type": "string",
                        "description": "",
                    },
                    "price": {
                        "type": "number",
                        "description": "",
                    },
                },
            },
            "required": [
                "name",
                "description",
                "number_of_items",
                "address",
                "price",
            ],
        }
    }