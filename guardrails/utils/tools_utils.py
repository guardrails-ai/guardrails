from typing import List, Optional

from guardrails.classes.schema.processed_schema import ProcessedSchema


# takes processed schema and converts it to a openai tool object
def schema_to_tool(schema) -> dict:
    tool = {
        "type": "function",
        "function": {
            "name": "gd_response_tool",
            "description": "A tool for generating responses to guardrails."
            " It must be called last in every response.",
            "parameters": schema,
            "required": schema["required"] or [],
        },
    }
    return tool


def json_function_calling_tool(
    schema: ProcessedSchema,
    tools: Optional[List] = None,
) -> List:
    tools = tools or []
    tools.append(schema_to_tool(schema))  # type: ignore
    return tools
