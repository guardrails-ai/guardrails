"""Rail class."""
import warnings
from dataclasses import dataclass
from typing import Optional, Sequence, Type

from lxml import etree as ET
from pydantic import BaseModel

from guardrails.prompt import Instructions, Prompt
from guardrails.schema import JsonSchema, Schema, StringSchema
from guardrails.utils.xml_utils import cast_xml_to_string
from guardrails.validator_base import ValidatorSpec

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

    prompt_schema: Optional[StringSchema]
    """The schema for the prompt.

    If None, the prompt is not validated.
    """

    instructions_schema: Optional[StringSchema]
    """The schema for the instructions.

    If None, the instructions are not validated.
    """

    msg_history_schema: Optional[StringSchema]
    """The schema for the message history.

    If None, the message history is not validated.
    """

    output_schema: Schema
    """The schema for the output."""

    instructions: Optional[Instructions]
    """The instructions to be passed to the LLM."""

    prompt: Optional[Prompt]
    """The prompt to be passed to the LLM."""

    version: str = "0.1"
    """The version of the RAIL file."""

    @property
    def output_type(self):
        """Returns the type of the output schema."""

        if isinstance(self.output_schema, StringSchema):
            return "str"
        else:
            return "dict"

    @classmethod
    def from_pydantic(
        cls,
        output_class: Type[BaseModel],
        prompt: Optional[str] = None,
        instructions: Optional[str] = None,
        reask_prompt: Optional[str] = None,
        reask_instructions: Optional[str] = None,
    ):
        """Initializes a RAIL from a Pydantic model."""
        output_schema = cls.load_json_schema_from_pydantic(
            output_class,
            reask_prompt_template=reask_prompt,
            reask_instructions_template=reask_instructions,
        )

        return cls(
            prompt_schema=None,
            instructions_schema=None,
            msg_history_schema=None,
            output_schema=output_schema,
            instructions=cls.load_instructions(instructions, output_schema),
            prompt=cls.load_prompt(prompt, output_schema),
        )

    @classmethod
    def from_file(cls, file_path: str) -> "Rail":
        """Loads a RAIL from a file."""

        with open(file_path, "r") as f:
            xml = f.read()
        return cls.from_string(xml)

    @classmethod
    def from_string(cls, string: str) -> "Rail":
        """Initializes a RAIL from a string."""

        return cls.from_xml(ET.fromstring(string, parser=XMLPARSER))

    @classmethod
    def from_xml(cls, xml: ET._Element):
        """Initializes a RAIL from an XML tree."""

        if "version" not in xml.attrib or xml.attrib["version"] != "0.1":
            raise ValueError(
                "RAIL file must have a version attribute set to 0.1."
                "Change the opening <rail> element to: <rail version='0.1'>."
            )

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
        output_schema = cls.load_output_schema_from_xml(
            raw_output_schema,
            reask_prompt=reask_prompt,
            reask_instructions=reask_instructions,
        )

        # Parse instructions for the LLM. These are optional but if given,
        # LLMs can use them to improve their output. Commonly these are
        # prepended to the prompt.
        instructions_tag = xml.find("instructions")
        if instructions_tag is None:
            instructions = None
            instructions_schema = None
        else:
            instructions = cls.load_instructions(instructions_tag.text, output_schema)
            instructions_schema = cls.load_input_schema_from_xml(instructions_tag)

        # Load <prompt />
        prompt_tag = xml.find("prompt")
        if prompt_tag is None:
            warnings.warn("Prompt must be provided during __call__.")
            prompt = None
            prompt_schema = None
        else:
            prompt = cls.load_prompt(prompt_tag.text, output_schema)
            prompt_schema = cls.load_input_schema_from_xml(prompt_tag)

        # Get version
        version = xml.attrib["version"]
        version = cast_xml_to_string(version)

        return cls(
            prompt_schema=prompt_schema,
            instructions_schema=instructions_schema,
            msg_history_schema=None,
            output_schema=output_schema,
            instructions=instructions,
            prompt=prompt,
            version=version,
        )

    @classmethod
    def from_string_validators(
        cls,
        validators: Sequence[ValidatorSpec],
        description: Optional[str] = None,
        prompt: Optional[str] = None,
        instructions: Optional[str] = None,
        reask_prompt: Optional[str] = None,
        reask_instructions: Optional[str] = None,
    ):
        """Initializes a RAIL from a list of validators."""
        output_schema = cls.load_string_schema_from_string(
            validators,
            description=description,
            reask_prompt_template=reask_prompt,
            reask_instructions_template=reask_instructions,
        )

        return cls(
            prompt_schema=None,
            instructions_schema=None,
            msg_history_schema=None,
            output_schema=output_schema,
            instructions=cls.load_instructions(instructions, output_schema),
            prompt=cls.load_prompt(prompt, output_schema),
        )

    @staticmethod
    def load_input_schema_from_xml(
        root: Optional[ET._Element],
    ) -> Optional[StringSchema]:
        """Given the RAIL <input> element, create a Schema object."""

        if root is None or all(
            tag not in root.attrib for tag in ["format", "validators"]
        ):
            return None
        return StringSchema.from_xml(root)

    @staticmethod
    def load_output_schema_from_xml(
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
            return StringSchema.from_xml(
                root,
                reask_prompt_template=reask_prompt,
                reask_instructions_template=reask_instructions,
            )
        return JsonSchema.from_xml(
            root,
            reask_prompt_template=reask_prompt,
            reask_instructions_template=reask_instructions,
        )

    @staticmethod
    def load_string_schema_from_string(
        validators: Sequence[ValidatorSpec],
        description: Optional[str] = None,
        reask_prompt_template: Optional[str] = None,
        reask_instructions_template: Optional[str] = None,
    ):
        """Initializes a StringSchema using a list of validators."""
        return StringSchema.from_string(
            validators,
            description=description,
            reask_prompt_template=reask_prompt_template,
            reask_instructions_template=reask_instructions_template,
        )

    @staticmethod
    def load_json_schema_from_pydantic(
        output_class: Type[BaseModel],
        reask_prompt_template: Optional[str] = None,
        reask_instructions_template: Optional[str] = None,
    ):
        """Initializes a JsonSchema using a Pydantic model."""

        return JsonSchema.from_pydantic(
            output_class,
            reask_prompt_template=reask_prompt_template,
            reask_instructions_template=reask_instructions_template,
        )

    @staticmethod
    def load_instructions(
        text: Optional[str], output_schema: Schema
    ) -> Optional[Instructions]:
        """Given the RAIL <instructions> element, create Instructions."""
        if text is None:
            return None
        return Instructions(
            source=text or "",
            output_schema=output_schema.transpile(),
        )

    @staticmethod
    def load_prompt(text: Optional[str], output_schema: Schema) -> Optional[Prompt]:
        """Given the RAIL <prompt> element, create a Prompt object."""
        if text is None:
            return None
        return Prompt(
            source=text or "",
            output_schema=output_schema.transpile(),
        )
