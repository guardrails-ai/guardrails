from typing import (
    Optional,
)

from guardrails.classes.schema.processed_schema import ProcessedSchema

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