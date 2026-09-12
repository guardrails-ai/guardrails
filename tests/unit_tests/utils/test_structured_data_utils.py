from jsonschema import Draft202012Validator
from pydantic import BaseModel, Field, create_model
from typing import List

import pytest

from guardrails.schema.pydantic_schema import pydantic_model_to_schema

from guardrails.utils.structured_data_utils import (
    json_function_calling_tool,
    schema_to_tool,
    output_format_json_schema,
    set_additional_properties_false_iteratively,
)


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


def test_json_function_calling_tool():
    schema = pydantic_model_to_schema(Person)
    tools = json_function_calling_tool(schema.json_schema)
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


def test_output_format_json_schema():
    schema = output_format_json_schema(Schedule)
    assert schema == {
        "type": "json_schema",
        "json_schema": {
            "name": "Schedule",
            "schema": {
                "additionalProperties": False,
                "$defs": {
                    "Delivery": {
                        "additionalProperties": False,
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
            "strict": True,
        },
    }


@pytest.mark.parametrize(
    "field_name", ["minimum", "maximum", "default", "properties", "required", "type"]
)
def test_output_format_json_schema_preserves_keyword_field_names(field_name):
    model = create_model(
        "Payload", **{field_name: (int, Field(default=1, ge=0, le=10))}
    )

    schema = output_format_json_schema(model)["json_schema"]["schema"]

    assert set(schema["properties"]) == {field_name}
    assert schema["required"] == [field_name]
    assert schema["additionalProperties"] is False
    assert schema["properties"][field_name] == {
        "title": field_name.title(),
        "type": "integer",
    }
    Draft202012Validator.check_schema(schema)
    Draft202012Validator(schema).validate({field_name: 5})


@pytest.mark.parametrize("model_name", ["minimum", "maximum", "default", "properties"])
def test_output_format_json_schema_preserves_keyword_definition_names(model_name):
    nested_model = create_model(model_name, value=(int, Field(default=1, ge=0, le=10)))
    model = create_model("Payload", entry=(nested_model, ...))

    schema = output_format_json_schema(model)["json_schema"]["schema"]

    assert set(schema["$defs"]) == {model_name}
    assert schema["$defs"][model_name] == {
        "title": model_name,
        "type": "object",
        "properties": {"value": {"title": "Value", "type": "integer"}},
        "required": ["value"],
        "additionalProperties": False,
    }
    Draft202012Validator.check_schema(schema)
    Draft202012Validator(schema).validate({"entry": {"value": 5}})


def test_set_additional_properties_preserves_non_schema_definition_values():
    schema = {
        "type": "object",
        "properties": {"$defs": {"type": "string"}},
        "examples": [{"$defs": "literal"}],
    }
    Draft202012Validator.check_schema(schema)

    set_additional_properties_false_iteratively(schema)

    assert schema["examples"] == [{"$defs": "literal"}]
    Draft202012Validator.check_schema(schema)
    Draft202012Validator(schema).validate(schema["examples"][0])
