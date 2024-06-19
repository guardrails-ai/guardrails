from pydantic import BaseModel, Field
from typing import List

from guardrails.schema.pydantic_schema import pydantic_model_to_schema

from guardrails.utils.tools_utils import augment_tools_with_schema, schema_to_tool


class Delivery(BaseModel):
    customer: str = Field(description="customer name")
    pickup_time: str = Field(description="date and time of pickup")
    pickup_location: str = Field(description="address of pickup")
    dropoff_time: str = Field(description="date and time of dropoff")
    dropoff_location: str = Field(description="address of dropoff")
    price: str = Field(description="price of delivery with currency symbol included")
    items: str = Field(
        description="items for pickup/delivery typically"
        " something a single person can carry on a bike",
    )
    number_items: int = Field(description="number of items")


class Schedule(BaseModel):
    deliveries: List[Delivery] = Field(description="deliveries for messenger")


class Person(BaseModel):
    name: str
    age: int
    hair_color: str


def test_pydantic_model_to_schema():
    schema = pydantic_model_to_schema(Schedule)
    tool = schema_to_tool(schema.json_schema)
    assert tool == {
        "type": "function",
        "function": {
            "name": "gd_response_tool",
            "description": "A tool for generating responses to guardrails."
            " It must be called last in every response.",
            "parameters": {
                "$defs": {
                    "Delivery": {
                        "properties": {
                            "customer": {
                                "description": "customer name",
                                "title": "Customer",
                                "type": "string",
                            },
                            "pickup_time": {
                                "description": "date and time of pickup",
                                "title": "Pickup Time",
                                "type": "string",
                            },
                            "pickup_location": {
                                "description": "address of pickup",
                                "title": "Pickup Location",
                                "type": "string",
                            },
                            "dropoff_time": {
                                "description": "date and time of dropoff",
                                "title": "Dropoff Time",
                                "type": "string",
                            },
                            "dropoff_location": {
                                "description": "address of dropoff",
                                "title": "Dropoff Location",
                                "type": "string",
                            },
                            "price": {
                                "description": "price of delivery with"
                                " currency symbol included",
                                "title": "Price",
                                "type": "string",
                            },
                            "items": {
                                "description": "items for pickup/delivery typically"
                                " something a single person can carry on a bike",
                                "title": "Items",
                                "type": "string",
                            },
                            "number_items": {
                                "description": "number of items",
                                "title": "Number Items",
                                "type": "integer",
                            },
                        },
                        "required": [
                            "customer",
                            "pickup_time",
                            "pickup_location",
                            "dropoff_time",
                            "dropoff_location",
                            "price",
                            "items",
                            "number_items",
                        ],
                        "title": "Delivery",
                        "type": "object",
                    }
                },
                "properties": {
                    "deliveries": {
                        "description": "deliveries for messenger",
                        "items": {"$ref": "#/$defs/Delivery"},
                        "title": "Deliveries",
                        "type": "array",
                    }
                },
                "required": ["deliveries"],
                "title": "Schedule",
                "type": "object",
            },
            "required": ["deliveries"],
        },
    }


def test_augment_tools_with_schema():
    schema = pydantic_model_to_schema(Person)
    tools = augment_tools_with_schema(schema.json_schema)
    assert tools == [
        {
            "type": "function",
            "function": {
                "name": "gd_response_tool",
                "description": "A tool for generating responses to guardrails."
                " It must be called last in every response.",
                "parameters": {
                    "properties": {
                        "name": {"title": "Name", "type": "string"},
                        "age": {"title": "Age", "type": "integer"},
                        "hair_color": {"title": "Hair Color", "type": "string"},
                    },
                    "required": ["name", "age", "hair_color"],
                    "title": "Person",
                    "type": "object",
                },
                "required": ["name", "age", "hair_color"],
            },
        }
    ]
