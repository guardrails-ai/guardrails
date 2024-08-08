from typing import List, Optional
from guardrails.logger import logger
from guardrails.classes.schema.processed_schema import ProcessedSchema
from guardrails.types.pydantic import ModelOrListOfModels


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


def set_additional_properties_false_iteratively(schema):
    stack = [schema]
    while stack:
        current = stack.pop()
        if isinstance(current, dict):
            if "properties" in current:
                current["required"] = list(
                    current["properties"].keys()
                )  # this has to be set
            if "maximum" in current:
                logger.warn("Property maximum is not supported." " Dropping")
                current.pop("maximum")  # the api does not like these set
            if "minimum" in current:
                logger.warn("Property maximum is not supported." " Dropping")
                current.pop("minimum")  # the api does not like these set
            if "default" in current:
                logger.warn("Property default is not supported. Marking field Required")
                current.pop("default")  # the api does not like these set
            for prop in current.values():
                stack.append(prop)
        elif isinstance(current, list):
            for prop in current:
                stack.append(prop)
        if (
            isinstance(current, dict)
            and "additionalProperties" not in current
            and "type" in current
            and current["type"] == "object"
        ):
            current["additionalProperties"] = False  # the api needs these set


def json_function_calling_tool(
    schema: ProcessedSchema,
    tools: Optional[List] = None,
) -> List:
    tools = tools or []
    tools.append(schema_to_tool(schema))  # type: ignore
    return tools


def output_format_json_schema(schema: ModelOrListOfModels) -> dict:
    parsed_schema = schema.model_json_schema()  # type: ignore

    set_additional_properties_false_iteratively(parsed_schema)

    return {
        "type": "json_schema",
        "json_schema": {
            "name": parsed_schema["title"],
            "schema": parsed_schema,
            "strict": True,
        },  # type: ignore
    }
