"""Rail class."""
from dataclasses import dataclass, field
from typing import List, Optional

from lxml import etree as ET

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
                        expressions.append(body[start + 1 : i])
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
