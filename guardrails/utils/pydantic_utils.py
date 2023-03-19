"""
RAIL File:

Version A:
    <pydantic 
        name="person" 
        description="Information about a person." 
        model="Person" 
        on-fail-{root-validator-name}="blah" 
        # on-fail-{field-name}-{validator-name}="blah" 
        on-fail-age-between-0-150="reask" 
    />
where Person is a Pydantic BaseModel.

    <pydantic 
        model="Person" 
        name="person" 
        description="Information about a person." 
        on-fail-{root-validator-name}="blah" 
    >
        # Omit fields that don't need a custom description or on-fail
        <field name="age" description="The age of the person." on-fail-between-0-150="reask" />
    </pydantic>

Which is converted to:

    <object name="person" description="Information about a person." model="Person">
        <string name="name" description="The name of the person." on-fail-{validator-name}="blah" />
        <integer name="age" description="The age of the person." on-fail-between-0-150="reask" />
    </object>

internally.
"""
from typing import TYPE_CHECKING, Dict

import lxml
from griffe.dataclasses import Docstring
from griffe.docstrings.parsers import Parser, parse

if TYPE_CHECKING:
    from pydantic import BaseModel


def get_field_descriptions(model: "BaseModel") -> Dict[str, str]:
    """Get the descriptions of the fields in a Pydantic model using the docstring."""
    docstring = Docstring(model.__doc__, lineno=1)
    parsed = parse(docstring, Parser.google)

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


def convert_field_element(element: lxml.etree._Element) -> lxml.etree._Element:
    """Convert a Pydantic BaseModel to an XML schema."""

    # Get the following attributes
    # TODO: add on-fail
    name = element.attrib["name"]
    description = element.attrib["description"]
    pydantic_cls = element.attrib["pydantic"]

    # Get the Pydantic model
    M = pydantic_models[pydantic_cls]
    schema = M.schema()
    field_descriptions = get_field_descriptions(M)

    # Make the XML as follows using lxml
    # <object name="..." description="..." format="semicolon separated root validators" pydantic="ModelName">
    #     <type name="..." description="..." format="semicolon separated validators" />
    # </object>

    # Add the object element, opening tag
    xml = ""
    root_validators = "; ".join(list(pydantic_validators[M]["__root__"].keys()))
    xml += f'<object name="{name}"'
    if description:
        xml += f' description="{description}"'
    if root_validators:
        xml += f' format="{root_validators}"'
    xml += f' pydantic="{pydantic_cls}"'
    xml += ">"

    # Add all the nested fields
    for field in schema["properties"]:
        properties = schema["properties"][field]
        field_type = PYDANTIC_SCHEMA_TYPE_MAP[properties["type"]]
        field_validators = "; ".join(list(pydantic_validators[M][field].keys()))
        try:
            field_description = field_descriptions[field]
        except KeyError:
            field_description = ""
        xml += f"<{field_type}"
        xml += f' name="{field}"'
        if field_description:
            xml += f' description="{field_descriptions[field]}"'
        if field_validators:
            xml += f' format="{field_validators}"'
        xml += f" />"

    # Close the object element
    xml += "</object>"

    # Convert the XML to a lxml.etree._Element
    return lxml.etree.fromstring(xml)


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
