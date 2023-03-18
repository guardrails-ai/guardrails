"""Rail class."""
from copy import deepcopy
from dataclasses import dataclass
from typing import Optional

from lxml import etree as ET

from guardrails.prompt import Prompt
from guardrails.schema import InputSchema, OutputSchema, Schema
from guardrails.utils.reask_utils import extract_prompt_from_xml

XMLPARSER = ET.XMLParser(encoding="utf-8")


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
    prompt: Optional[Prompt] = (None,)
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
        script = xml.find("script")
        if script is not None:
            cls.load_script(script)

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
        output_schema = cls.load_output_schema(raw_output_schema)

        # Load <prompt />
        prompt = xml.find("prompt")
        if prompt is None:
            raise ValueError("RAIL file must contain a prompt element.")
        prompt = cls.load_prompt(prompt, output_schema)

        return cls(
            input_schema=input_schema,
            output_schema=output_schema,
            prompt=prompt,
            version=xml.attrib["version"],
        )

    @staticmethod
    def load_schema(root: ET._Element) -> Schema:
        """Given the RAIL <input> or <output> element, create a Schema
        object."""
        output = Schema(parsed_rail=root)

        return output

    @staticmethod
    def load_input_schema(root: ET._Element) -> InputSchema:
        """Given the RAIL <input> element, create a Schema object."""
        # Recast the schema as an InputSchema.
        return InputSchema.from_schema(Rail.load_schema(root))

    @staticmethod
    def load_output_schema(root: ET._Element) -> OutputSchema:
        """Given the RAIL <output> element, create a Schema object."""
        # Recast the schema as an OutputSchema.
        return OutputSchema.from_schema(Rail.load_schema(root))

    @staticmethod
    def load_prompt(root: ET._Element, output_schema: OutputSchema) -> Prompt:
        """Given the RAIL <prompt> element, create a Prompt object."""
        text = root.text
        output_schema_prompt = extract_prompt_from_xml(
            deepcopy(output_schema.parsed_rail)
        )

        return Prompt(text, output_schema=output_schema_prompt)

    @staticmethod
    def load_script(root: ET._Element) -> None:
        """Given the RAIL <script> element, load and execute the script."""
        if "language" not in root.attrib:
            raise ValueError("Script element must have a language attribute.")

        language = root.attrib["language"]
        if language != "python":
            raise ValueError("Only python scripts are supported right now.")

        exec(root.text, globals())
