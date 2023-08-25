"""Rail class."""
import warnings
from dataclasses import dataclass
from typing import List, Optional, Type

from lxml import etree as ET
from lxml.etree import Element, SubElement
from pydantic import BaseModel

from guardrails.prompt import Instructions, Prompt
from guardrails.schema import JsonSchema, Schema, StringSchema
from guardrails.utils.pydantic_utils import (
    attach_validators_to_element,
    create_xml_element_for_base_model,
)
from guardrails.validators import Validator

# TODO: Logging
XMLPARSER = ET.XMLParser(encoding="utf-8")


@dataclass
class Rail:
    """RAIL (Reliable AI Language) is a dialect of XML that allows users to
    specify guardrails for large language models (LLMs).

    A RAIL file contains a root element called
        `<rail version="x.y">`
    that contains the following elements as children:
        1. `<input strict=True/False>`, which contains the input schema
        2. `<output strict=True/False>`, which contains the output schema
        3. `<prompt>`, which contains the prompt to be passed to the LLM
        4. `<instructions>`, which contains the instructions to be passed to the LLM
    """

    input_schema: Optional[Schema] = (None,)
    output_schema: Optional[Schema] = (None,)
    instructions: Optional[Instructions] = (None,)
    prompt: Optional[Prompt] = (None,)
    version: Optional[str] = ("0.1",)

    @classmethod
    def from_pydantic(
        cls,
        output_class: BaseModel,
        prompt: Optional[str] = None,
        instructions: Optional[str] = None,
        reask_prompt: Optional[str] = None,
        reask_instructions: Optional[str] = None,
    ):
        xml = generate_xml_code(
            output_class=output_class,
            prompt=prompt,
            instructions=instructions,
            reask_prompt=reask_prompt,
            reask_instructions=reask_instructions,
        )
        return cls.from_xml(xml)

    @classmethod
    def from_file(cls, file_path: str) -> "Rail":
        with open(file_path, "r") as f:
            xml = f.read()
        return cls.from_string(xml)

    @classmethod
    def from_string(cls, string: str) -> "Rail":
        return cls.from_xml(ET.fromstring(string, parser=XMLPARSER))

    @classmethod
    def from_xml(cls, xml: ET._Element):
        if "version" not in xml.attrib or xml.attrib["version"] != "0.1":
            raise ValueError(
                "RAIL file must have a version attribute set to 0.1."
                "Change the opening <rail> element to: <rail version='0.1'>."
            )

        # Load <input /> schema
        raw_input_schema = xml.find("input")
        if raw_input_schema is None:
            # No input schema, so do no input checking.
            input_schema = Schema()
        else:
            input_schema = cls.load_input_schema(raw_input_schema)

        # Load <output /> schema
        raw_output_schema = xml.find("output")
        if raw_output_schema is None:
            raise ValueError("RAIL file must contain a output-schema element.")
        raw_output_schema = ET.tostring(raw_output_schema, encoding="utf-8")
        raw_output_schema = ET.fromstring(raw_output_schema, parser=XMLPARSER)
        # If reasking prompt and instructions are provided, add them to the schema.
        reask_prompt = xml.find("reask_prompt")
        if reask_prompt is not None:
            reask_prompt = reask_prompt.text
        reask_instructions = xml.find("reask_instructions")
        if reask_instructions is not None:
            reask_instructions = reask_instructions.text
        output_schema = cls.load_output_schema(
            raw_output_schema,
            reask_prompt=reask_prompt,
            reask_instructions=reask_instructions,
        )

        # Parse instructions for the LLM. These are optional but if given,
        # LLMs can use them to improve their output. Commonly these are
        # prepended to the prompt.
        instructions = xml.find("instructions")
        if instructions is not None:
            instructions = cls.load_instructions(instructions, output_schema)

        # Load <prompt />
        prompt = xml.find("prompt")
        if prompt is None:
            warnings.warn("Prompt must be provided during __call__.")
        else:
            prompt = cls.load_prompt(prompt, output_schema)

        return cls(
            input_schema=input_schema,
            output_schema=output_schema,
            instructions=instructions,
            prompt=prompt,
            version=xml.attrib["version"],
        )

    @classmethod
    def from_string_validators(
        cls,
        validators: List[Validator],
        description: Optional[str] = None,
        prompt: Optional[str] = None,
        instructions: Optional[str] = None,
        reask_prompt: Optional[str] = None,
        reask_instructions: Optional[str] = None,
    ):
        xml = generate_xml_code(
            prompt=prompt,
            instructions=instructions,
            reask_prompt=reask_prompt,
            reask_instructions=reask_instructions,
            validators=validators,
            description=description,
        )
        return cls.from_xml(xml)

    @staticmethod
    def load_schema(root: ET._Element) -> Schema:
        """Given the RAIL <input> or <output> element, create a Schema
        object."""
        return Schema(root)

    @staticmethod
    def load_input_schema(root: ET._Element) -> Schema:
        """Given the RAIL <input> element, create a Schema object."""
        # Recast the schema as an InputSchema.
        return Schema(root)

    @staticmethod
    def load_output_schema(
        root: ET._Element,
        reask_prompt: Optional[str] = None,
        reask_instructions: Optional[str] = None,
    ) -> Schema:
        """Given the RAIL <output> element, create a Schema object.

        Args:
            root: The root element of the output schema.
            reask_prompt: If provided, the prompt when reasking the LLM.
            reask_instructions: If provided, the instructions when reasking the LLM.

        Returns:
            A Schema object.
        """
        # If root contains a `type="string"` attribute, then it's a StringSchema
        if "type" in root.attrib and root.attrib["type"] == "string":
            return StringSchema(
                root,
                reask_prompt_template=reask_prompt,
                reask_instructions_template=reask_instructions,
            )
        return JsonSchema(
            root,
            reask_prompt_template=reask_prompt,
            reask_instructions_template=reask_instructions,
        )

    @staticmethod
    def load_instructions(root: ET._Element, output_schema: Schema) -> Instructions:
        """Given the RAIL <instructions> element, create Instructions."""
        return Instructions(
            source=root.text,
            output_schema=output_schema.transpile(),
        )

    @staticmethod
    def load_prompt(root: ET._Element, output_schema: Schema) -> Prompt:
        """Given the RAIL <prompt> element, create a Prompt object."""
        return Prompt(
            source=root.text,
            output_schema=output_schema.transpile(),
        )


def generate_xml_code(
    prompt: str,
    output_class: Optional[Type[BaseModel]] = None,
    instructions: Optional[str] = None,
    reask_prompt: Optional[str] = None,
    reask_instructions: Optional[str] = None,
    validators: Optional[List[Validator]] = None,
    description: Optional[str] = None,
) -> ET._Element:
    """Generate XML RAIL Spec from a pydantic model and a prompt.

    Parameters: Arguments:
        prompt (str): The prompt for this RAIL spec.
        output_class (BaseModel, optional): The Pydantic model that represents the desired output schema.  Do not specify if using a string schema. Defaults to None.
        instructions (str, optional): Instructions for chat models. Defaults to None.
        reask_prompt (str, optional): An alternative prompt to use during reasks. Defaults to None.
        reask_instructions (str, optional): Alternative instructions to use during reasks. Defaults to None.
        validators (List[Validator], optional): The list of validators to apply to the string schema. Do not specify if using a Pydantic model. Defaults to None.
        description (str, optional): The description for a string schema. Do not specify if using a Pydantic model. Defaults to None.
    """  # noqa

    # Create the root element
    root = Element("rail")
    root.set("version", "0.1")

    # Create the output element
    output_element = SubElement(root, "output")

    if output_class and validators:
        warnings.warn(
            "Do not specify root level validators on a Pydantic model."
            "  These validators will be ignored."
        )

    if output_class is not None:
        # Create XML elements for the output_class
        create_xml_element_for_base_model(output_class, output_element)
    else:
        attach_validators_to_element(output_element, validators)
        output_element.set("description", description)
        output_element.set("type", "string")

    if prompt is not None:
        # Create the prompt element
        prompt_element = SubElement(root, "prompt")
        prompt_text = f"{prompt}"
        prompt_element.text = prompt_text

    if instructions is not None:
        # Create the instructions element
        instructions_element = SubElement(root, "instructions")
        instructions_text = f"{instructions}"
        instructions_element.text = instructions_text

    if reask_prompt is not None:
        # Create the reask_prompt element
        reask_prompt_element = SubElement(root, "reask_prompt")
        reask_prompt_text = f"{reask_prompt}"
        reask_prompt_element.text = reask_prompt_text

    if reask_instructions is not None:
        # Create the reask_instructions element
        reask_instructions_element = SubElement(root, "reask_instructions")
        reask_instructions_text = f"{reask_instructions}"
        reask_instructions_element.text = reask_instructions_text

    return root
