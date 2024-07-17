from typing import (
    Dict,
    List,
    Type,
    Union,
    cast,
    get_args,
    get_origin,
)

from pydantic import BaseModel

from guardrails.utils.safe_get import safe_get


def convert_pydantic_model_to_openai_fn(
    model: Union[Type[BaseModel], Type[List[Type[BaseModel]]]],
) -> Dict:
    """Convert a Pydantic BaseModel to an OpenAI function.

    Args:
        model: The Pydantic BaseModel to convert.

    Returns:
        OpenAI function paramters.
    """

    schema_model = model

    type_origin = get_origin(model)
    if type_origin == list:
        item_types = get_args(model)
        if len(item_types) > 1:
            raise ValueError("List data type must have exactly one child.")
        # No List[List] support; we've already declared that in our types
        schema_model = safe_get(item_types, 0)

    schema_model = cast(Type[BaseModel], schema_model)

    # Convert Pydantic model to JSON schema
    json_schema = schema_model.model_json_schema()
    json_schema["title"] = schema_model.__name__

    if type_origin == list:
        json_schema = {
            "title": f"Array<{json_schema.get('title')}>",
            "type": "array",
            "items": json_schema,
        }

    # Create OpenAI function parameters
    fn_params = {
        "name": json_schema["title"],
        "parameters": json_schema,
    }
    if "description" in json_schema and json_schema["description"] is not None:
        fn_params["description"] = json_schema["description"]

    # TODO: Update this to tools
    # Wrap in { "type": "function", "function": fn_params}
    return fn_params
