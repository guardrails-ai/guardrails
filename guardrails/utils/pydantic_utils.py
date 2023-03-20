"""Utilities for working with Pydantic models.

Guardrails lets users specify

<pydantic
    model="Person"
    name="person"
    description="Information about a person."
    on-fail-pydantic="reask" / "refrain" / "raise"
/>
"""
import logging
from typing import TYPE_CHECKING, Dict

from griffe.dataclasses import Docstring
from griffe.docstrings.parsers import Parser, parse

griffe_docstrings_google_logger = logging.getLogger("griffe.docstrings.google")
griffe_agents_nodes_logger = logging.getLogger("griffe.agents.nodes")

if TYPE_CHECKING:
    from pydantic import BaseModel


def get_field_descriptions(model: "BaseModel") -> Dict[str, str]:
    """Get the descriptions of the fields in a Pydantic model using the
    docstring."""
    griffe_docstrings_google_logger.disabled = True
    griffe_agents_nodes_logger.disabled = True
    try:
        docstring = Docstring(model.__doc__, lineno=1)
    except AttributeError:
        return {}
    parsed = parse(docstring, Parser.google)
    griffe_docstrings_google_logger.disabled = False
    griffe_agents_nodes_logger.disabled = False

    # TODO: change parsed[1] to an isinstance check for the args section
    return {
        field.name: field.description.replace("\n", " ")
        for field in parsed[1].as_dict()["value"]
    }


PYDANTIC_SCHEMA_TYPE_MAP = {
    "string": "string",
    "number": "float",
    "integer": "integer",
    "boolean": "bool",
    "object": "object",
    "array": "list",
}

pydantic_validators = {}
pydantic_models = {}


# Create a class decorator to register all the validators in a BaseModel
def register_pydantic(cls: type):
    """
    Register a Pydantic BaseModel. This is a class decorator that can
    be used in the following way:

    ```
    @register_pydantic
    class MyModel(BaseModel):
        ...
    ```

    This decorator does the following:
        1. Add the model to the pydantic_models dictionary.
        2. Register all pre and post validators.
        3. Register all pre and post root validators.
    """
    # Register the model
    pydantic_models[cls.__name__] = cls

    # Create a dictionary to store all the validators
    pydantic_validators[cls] = {}
    # All all pre and post validators, for each field in the model
    for field in cls.__fields__.values():
        pydantic_validators[cls][field.name] = {}
        if field.pre_validators:
            for validator in field.pre_validators:
                pydantic_validators[cls][field.name][
                    validator.func_name.replace("_", "-")
                ] = validator
        if field.post_validators:
            for validator in field.post_validators:
                pydantic_validators[cls][field.name][
                    validator.func_name.replace("_", "-")
                ] = validator

    pydantic_validators[cls]["__root__"] = {}
    # Add all pre and post root validators
    if cls.__pre_root_validators__:
        for _, validator in cls.__pre_root_validators__:
            pydantic_validators[cls]["__root__"][
                validator.__name__.replace("_", "-")
            ] = validator

    if cls.__post_root_validators__:
        for _, validator in cls.__post_root_validators__:
            pydantic_validators[cls]["__root__"][
                validator.__name__.replace("_", "-")
            ] = validator
    return cls
