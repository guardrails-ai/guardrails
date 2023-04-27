"""Rail class."""
import typing
from dataclasses import dataclass, field
from datetime import date, time
from typing import List, Optional, Type

from lxml import etree as ET
from lxml.etree import Element, SubElement, tostring
from pydantic import BaseModel, HttpUrl

from guardrails.prompt import Instructions, Prompt
from guardrails.schema import InputSchema, OutputSchema, Schema

# TODO: Logging
XMLPARSER = ET.XMLParser(encoding="utf-8")


@dataclass
class Script:
    variables: dict = field(default_factory=dict)
    language: str = "python"

    @classmethod
    def from_xml(cls, root: ET._Element) -> "Script":
        if "language" not in root.attrib:
            raise ValueError("Script element must have a language attribute.")

        language = root.attrib["language"]
        if language != "python":
            raise ValueError("Only python scripts are supported right now.")

        # Run the script in the global namespace, returning the additional
        # globals that were created.
        keys = set(globals().keys())
        exec(root.text, globals())
        new_keys = globals().keys()
        variables = {k: globals()[k] for k in new_keys if k not in keys}
        return cls(variables, language)

    @staticmethod
    def find_expressions(body) -> List[str]:
        """Get all expressions, written as {...} in a string body."""
        expressions = []
        stack = []
        start = -1

        for i, char in enumerate(body):
            if char == "{":
                if not stack:
                    start = i
                stack.append(char)
            elif char == "}":
                if stack and stack[-1] == "{":
                    stack.pop()
                    if not stack:
                        expressions.append(body[start + 1: i])
                else:
                    stack.append(char)
        return expressions

    def replace_expressions(self, body: str) -> str:
        """Replace all expressions in a string body with their evaluated
        values."""
        # Decode the body if it's a bytes object.
        if isinstance(body, bytes):
            body = body.decode("utf-8")
        for expr in self.find_expressions(body):
            # The replacement should be inserted as a Python expression, inside
            # curly braces.
            replacement = self(expr)
            # If a string, wrap it in '' quotes.
            if isinstance(replacement, str):
                replacement = f"'{replacement}'"
            body = body.replace(f"{{{expr}}}", f"{{{replacement}}}")

        return body

    def __call__(self, expr: str):
        """Eval expression in the script's namespace."""
        return eval(expr, {**globals(), **self.variables})


@dataclass
class Rail:
    """RAIL (Reliable AI Language) is a dialect of XML that allows users to
    specify guardrails for large language models (LLMs).

    A RAIL file contains a root element called
        `<rail version="x.y">`
    that contains the following elements as children:
        1. `<script language="python">`, which contains the script to be executed
        2. `<input strict=True/False>`, which contains the input schema
        3. `<output strict=True/False>`, which contains the output schema
        4. `<prompt>`, which contains the prompt to be passed to the LLM
    """

    input_schema: Optional[InputSchema] = (None,)
    output_schema: Optional[OutputSchema] = (None,)
    instructions: Optional[Instructions] = (None,)
    prompt: Optional[Prompt] = (None,)
    script: Optional[Script] = (None,)
    version: Optional[str] = ("0.1",)

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

        # Execute the script before validating the rest of the RAIL file.
        raw_script = xml.find("script")
        if raw_script is not None:
            script = cls.load_script(raw_script)
        else:
            script = Script()

        # Load <input /> schema
        raw_input_schema = xml.find("input")
        if raw_input_schema is None:
            # No input schema, so do no input checking.
            input_schema = InputSchema()
        else:
            input_schema = cls.load_input_schema(raw_input_schema)

        # Load <output /> schema
        raw_output_schema = xml.find("output")
        if raw_output_schema is None:
            raise ValueError("RAIL file must contain a output-schema element.")
        # Replace all expressions in the <output /> schema.
        raw_output_schema = script.replace_expressions(ET.tostring(raw_output_schema))
        raw_output_schema = ET.fromstring(raw_output_schema, parser=XMLPARSER)
        output_schema = cls.load_output_schema(raw_output_schema)
        # Parse instructions for the LLM. These are optional but if given,
        # LLMs can use them to improve their output. Commonly these are
        # prepended to the prompt.
        instructions = xml.find("instructions")
        if instructions is not None:
            instructions = cls.load_instructions(instructions, output_schema)

        # Load <prompt />
        prompt = xml.find("prompt")
        if prompt is None:
            raise ValueError("RAIL file must contain a prompt element.")
        prompt = cls.load_prompt(prompt, output_schema)

        return cls(
            input_schema=input_schema,
            output_schema=output_schema,
            instructions=instructions,
            prompt=prompt,
            script=script,
            version=xml.attrib["version"],
        )

    @staticmethod
    def load_schema(root: ET._Element) -> Schema:
        """Given the RAIL <input> or <output> element, create a Schema
        object."""
        return Schema(root)

    @staticmethod
    def load_input_schema(root: ET._Element) -> InputSchema:
        """Given the RAIL <input> element, create a Schema object."""
        # Recast the schema as an InputSchema.
        return InputSchema(root)

    @staticmethod
    def load_output_schema(root: ET._Element) -> OutputSchema:
        """Given the RAIL <output> element, create a Schema object."""
        # Recast the schema as an OutputSchema.
        return OutputSchema(root)

    @staticmethod
    def load_instructions(
            root: ET._Element, output_schema: OutputSchema
    ) -> Instructions:
        """Given the RAIL <instructions> element, create Instructions."""
        return Instructions(
            source=root.text,
            output_schema=output_schema.transpile(),
        )

    @staticmethod
    def load_prompt(root: ET._Element, output_schema: OutputSchema) -> Prompt:
        """Given the RAIL <prompt> element, create a Prompt object."""
        return Prompt(
            source=root.text,
            output_schema=output_schema.transpile(),
        )

    @staticmethod
    def load_script(root: ET._Element) -> Script:
        """Given the RAIL <script> element, load and execute the script."""
        return Script.from_xml(root)

    @classmethod
    def from_pydantic(cls, output_class, prompt, instructions):
        xml = generate_xml_code(output_class, prompt, instructions)
        return cls.from_string(xml)


def create_xml_elements_for_model(parent_element, model, model_name):
    # Dictionary to keep track of choice elements and their corresponding case elements
    choice_elements: typing.Dict[str, Element] = {}

    # Iterate through all elements to detect any discriminator fields, and store the discriminator
    discriminators = set()  # List of discriminator fields (used in choices)
    for field_name, field_type in model.__fields__.items():
        if hasattr(field_type.field_info, "gd_if") and field_type.field_info.gd_if is not None:
            gd_if = field_type.field_info.gd_if
            discriminator = gd_if.split("==")[0].strip()
            discriminators.add(discriminator)

    # Iterate through the fields of the model and create elements for each field
    for field_name, field_type in model.__fields__.items():
        # Skip discriminator fields
        if field_name in discriminators:
            continue

        # Determine the XML element name based on the field type
        if field_type.type_ == bool:
            field_element_name = "bool"
        elif field_type.type_ == date:
            field_element_name = "date"
        elif field_type.type_ == float:
            field_element_name = "float"
        elif field_type.type_ == int:
            field_element_name = "integer"
        elif (
                issubclass(field_type.type_, List)
                or typing.get_origin(model.__annotations__.get(field_name)) == list
        ):
            field_element_name = "list"
            # Handle list of objects
            inner_type = field_type.type_
            if issubclass(inner_type, BaseModel):
                field_element = SubElement(parent_element, field_element_name)
                field_element.set("name", field_name)
                # Add the object element inside the list element
                object_element = SubElement(field_element, "object")
                create_xml_elements_for_model(
                    object_element, field_type.type_, field_name
                )
                continue
        elif issubclass(field_type.type_, BaseModel):
            field_element_name = "object"
        elif field_type.type_ == str:
            field_element_name = "string"
        elif field_type.type_ == time:
            field_element_name = "time"
        elif field_type.type_ == HttpUrl:
            field_element_name = "url"
        else:
            # Skip unsupported types
            # TODO: Add logging?
            continue

        # Check if the field has a discriminator
        if hasattr(field_type.field_info, "gd_if") and field_type.field_info.gd_if is not None:
            gd_if = field_type.field_info.gd_if

            # split gd_if on "==" to get the field name and value
            discriminator_field_name, discriminator_field_value = gd_if.split("==")

            # ensure both discriminator_field_name and discriminator_field_value are strings of len > 0 and that the
            # discriminator_field_name is a valid field
            if len(discriminator_field_name) == 0 or len(
                    discriminator_field_value) == 0 or discriminator_field_name not in model.__fields__:
                raise ValueError(f"Invalid gd_if for field {discriminator_field_name}")

            # Check if a choice element already exists for the discriminator_field_name
            if discriminator_field_name in choice_elements:
                choice_element = choice_elements[discriminator_field_name]
            else:
                choice_element = SubElement(parent_element, "choice")
                choice_element.set("name", discriminator_field_name)
                choice_elements[discriminator_field_name] = choice_element

            # Create the case element
            case_element = SubElement(choice_element, "case")
            case_element.set("name", discriminator_field_value)

            # Create the field element inside the case element
            field_element = SubElement(case_element, field_element_name)
            field_element.set("name", field_name)
        else:
            # Skip creating the field element if it's a discriminator field
            if field_name in choice_elements.values():
                continue

            # Create the field
            field_element = SubElement(parent_element, field_element_name)
            field_element.set("name", field_name)

        # Extract validators from the Pydantic field and add format and on-fail attributes
        if hasattr(field_type.field_info, "gd_validators") and field_type.field_info.gd_validators is not None:
            for validator in field_type.field_info.gd_validators:
                # Set the format attribute based on the validator class name
                format_prompt = validator.to_prompt(with_keywords=False)
                field_element.set("format", format_prompt)
                # Set the on-fail attribute based on the on_fail value
                on_fail_action = validator.on_fail.__name__ if validator.on_fail else "noop"
                field_element.set("on-fail-" + validator.rail_alias, on_fail_action)

        # Handle nested models
        if issubclass(field_type.type_, BaseModel):
            # If the field has a discriminator, use the discriminator field value as the model name
            if hasattr(field_type.field_info, "gd_if"):
                _, discriminator_field_value = field_type.field_info.gd_if.split("==")
                nested_model_name = discriminator_field_value
            else:
                nested_model_name = field_name
            create_xml_elements_for_model(field_element, field_type.type_, nested_model_name)


def generate_xml_code(output_class: Type[BaseModel], prompt: str, instructions: str) -> str:
    # Create the root element
    root = Element("rail")
    root.set("version", "0.1")

    # Create the output element
    output = SubElement(root, "output")

    # Create XML elements for the output_class
    create_xml_elements_for_model(output, output_class, output_class.__name__)

    # Create the prompt element
    prompt_element = SubElement(root, "prompt")
    prompt_text = f"\n{prompt}\n"
    prompt_text += "@complete_json_suffix_v2\n"
    prompt_element.text = prompt_text

    # Create the instructions element
    instructions_element = SubElement(root, "instructions")
    instructions_text = f"\n{instructions}\n"
    instructions_element.text = instructions_text

    # Convert the XML tree to a string
    xml_code = tostring(root, encoding="unicode", pretty_print=True)

    return xml_code
