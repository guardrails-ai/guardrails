from typing import (
    Any,
    Awaitable,
    Callable,
    Dict,
    Generic,
    Iterable,
    List,
    Optional,
    Sequence,
    Type,
    Union,
    cast,
    overload,
)

from guardrails.classes.schema.processed_schema import ProcessedSchema


def process_property(tool: dict, key: str, value: dict) -> dict:
    property = {
        "type": value["type"],
        "description": value.get("description", ""),
    }
    if value.get("format"):
        property["format"] = value["format"]
    if value.get("enum"):
        property["enum"] = value["enum"]
    if value.get("minimum"):
        property["minimum"] = value["minimum"]
    if value.get("maximum"):
        property["maximum"] = value["maximum"]
    if value.get("minLength"):
        property["minLength"] = value["minLength"]
    if value.get("maxLength"):
        property["maxLength"] = value["maxLength"]
    if value.get("pattern"):
        property["pattern"] = value["pattern"]
    if value.get("items"):
        property["items"] = process_property(tool, key, value["items"])
    if value.get("properties"):
        property["properties"] = {}
        for sub_key, sub_value in value["properties"].items():
            property["properties"][sub_key] = process_property(tool, sub_key, sub_value)
    return property

# takes processed schema and converts it to a openai tool object
def schema_to_tool(schema: ProcessedSchema) -> dict:
    json_schema = schema.json_schema
    tool = {
        "type": "function",
        "function": {
            "name": "gd_response_tool",
            "description": "A tool for generating responses to guardrails. It must be called last in every response.",
            "parameters": schema.json_schema,
            "required": json_schema["required"] or [],
        },
    }

    return tool

def augment_tools_with_schema(schema: ProcessedSchema, tools: Optional[list] = [],) -> list:
    tools.append(schema_to_tool(schema))
    return tools

def tools_prompt_string()-> str:
    return "Tools have been provided. Call the gd_response_tool with the response as the last thing you do."