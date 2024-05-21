from typing import Dict, List
from guardrails_api_client.models.validation_type import ValidationType
from lxml import etree as ET
from lxml.etree import _Element, XMLParser
from guardrails_api_client import ModelSchema, SimpleTypes, ValidatorReference
from guardrails.classes.execution.guard_execution_options import GuardExecutionOptions
from guardrails.classes.output_type import OutputTypes
from guardrails.classes.schema.processed_schema import ProcessedSchema
from guardrails.logger import logger
from guardrails.types import RailTypes
from guardrails.utils.regex_utils import split_on
from guardrails.utils.validator_utils import get_validator
from guardrails.utils.xml_utils import xml_to_string
from guardrails.validator_base import OnFailAction, Validator


STRING_TAGS = ["instructions", "prompt", "reask_instructions", "reask_prompt"]


def parse_on_fail_handlers(element: _Element) -> Dict[str, OnFailAction]:
    on_fail_handlers: Dict[str, OnFailAction] = {}
    for key, value in element.attrib.items():
        key = xml_to_string(key)
        if key.startswith("on-fail-"):
            on_fail_handler_name = key[len("on-fail-") :]
            on_fail_handler = OnFailAction(value)
            on_fail_handlers[on_fail_handler_name] = on_fail_handler
    return on_fail_handlers


def get_validators(element: _Element) -> List[Validator]:
    validators_string: str = xml_to_string(element.attrib.get("validators", ""))
    validator_specs = split_on(validators_string, ";")
    on_fail_handlers = parse_on_fail_handlers(element)
    validators: List[Validator] = []
    for v in validator_specs:
        validator: Validator = get_validator(v)
        if not validator:
            continue
        on_fail = on_fail_handlers.get(validator.rail_alias, OnFailAction.NOOP)
        validator.on_fail_descriptor = on_fail
        validators.append(validator)
    return validators


def extract_validators(
    element: _Element, processed_schema: ProcessedSchema, json_path: str
):
    validators = get_validators(element)
    for validator in validators:
        validator_reference = ValidatorReference(
            id=validator.rail_alias,
            on=json_path,
            onFail=validator.on_fail_descriptor,
            kwargs=validator.get_args(),
        )
        processed_schema.validators.append(validator_reference)

    if validators:
        path_validators = processed_schema.validator_map.get(json_path, [])
        path_validators.extend(validators)
        processed_schema.validator_map[json_path] = path_validators


def parse_element(
    element: _Element, processed_schema: ProcessedSchema, json_path: str = "$"
) -> ModelSchema:
    """
    Takes an XML element
    Extracts validators to add to the 'validators' list and validator_map
    Returns a ModelSchema
    """
    schema_type = element.tag
    if element.tag in STRING_TAGS:
        schema_type = RailTypes.STRING
    elif element.tag == "output":
        schema_type = element.attrib.get("type", RailTypes.OBJECT)

    description = xml_to_string(element.attrib.get("description"))

    # Extract validators from RAIL and assign into ProcessedSchema
    extract_validators(element, processed_schema, json_path)

    json_path = json_path.replace(".*", "")

    if schema_type == RailTypes.STRING:
        format = xml_to_string(element.attrib.get("format"))
        return ModelSchema(
            type=ValidationType(SimpleTypes.STRING),
            description=description,
            format=format,
        )
    elif schema_type == RailTypes.INTEGER:
        format = xml_to_string(element.attrib.get("format"))
        return ModelSchema(
            type=ValidationType(SimpleTypes.INTEGER),
            description=description,
            format=format,
        )
    elif schema_type == RailTypes.FLOAT:
        format = xml_to_string(element.attrib.get("format", RailTypes.FLOAT))
        return ModelSchema(
            type=ValidationType(SimpleTypes.NUMBER),
            description=description,
            format=format,
        )
    elif schema_type == RailTypes.BOOL:
        return ModelSchema(
            type=ValidationType(SimpleTypes.BOOLEAN), description=description
        )
    elif schema_type == RailTypes.DATE:
        format = xml_to_string(element.attrib.get("format", RailTypes.DATE))
        return ModelSchema(
            type=ValidationType(SimpleTypes.STRING),
            description=description,
            format=format,
        )
    elif schema_type == RailTypes.TIME:
        format = xml_to_string(element.attrib.get("format", RailTypes.TIME))
        return ModelSchema(
            type=ValidationType(SimpleTypes.STRING),
            description=description,
            format=format,
        )
    elif schema_type == RailTypes.DATETIME:
        format = xml_to_string(element.attrib.get("format", RailTypes.DATETIME))
        return ModelSchema(
            type=ValidationType(SimpleTypes.STRING),
            description=description,
            format=format,
        )
    elif schema_type == RailTypes.PERCENTAGE:
        format = xml_to_string(element.attrib.get("format", RailTypes.PERCENTAGE))
        return ModelSchema(
            type=ValidationType(SimpleTypes.STRING),
            description=description,
            format=format,
        )
    elif schema_type == RailTypes.ENUM:
        format = xml_to_string(element.attrib.get("format"))
        csv = xml_to_string(element.attrib.get("values", ""))
        values = list(map(lambda v: v.strip(), csv.split(",")))
        return ModelSchema(
            type=ValidationType(SimpleTypes.STRING),
            description=description,
            format=format,
            enum=values,
        )
    elif schema_type == RailTypes.LIST:
        items = None
        children = list(element)
        num_of_children = len(children)
        if num_of_children == 0 or num_of_children > 1:
            raise ValueError(
                "<list /> RAIL elements must have precisely 1 child element!"
            )
        first_child = children[0]
        child_schema = parse_element(first_child, processed_schema, f"{json_path}.*")
        items = child_schema.to_dict()
        return ModelSchema(
            type=ValidationType(SimpleTypes.ARRAY), items=items, description=description
        )
    elif schema_type == RailTypes.OBJECT:
        properties = {}
        required: List[str] = []
        for child in element:
            name = child.get("name")
            child_required = child.get("required") == "true"
            if not name:
                output_path = json_path.replace("$.", "output.")
                logger.warn(f"{output_path} has a nameless child which is not allowed!")
                continue
            if child_required:
                required.append(name)
            child_schema = parse_element(child, processed_schema, f"{json_path}.{name}")
            properties[name] = child_schema.to_dict()

        return ModelSchema(
            type=ValidationType(SimpleTypes.OBJECT),
            properties=properties,
            description=description,
            required=required,
        )
    elif schema_type == RailTypes.CHOICE:
        """
        Since our ModelSchema class reflects the pure JSON Schema structure
        this implementation of choice-case strays from the
        Discriminated Unions specification as defined 
        by OpenAPI that Pydantic uses.
        
        We should verify that LLM's understand this syntax properly.
        If they do not, we can manually add the 'discriminator' property to
        the schema after calling 'ModelSchema.to_dict()'.
        
        JSON Schema Conditional Subschemas
        https://json-schema.org/understanding-json-schema/reference/conditionals#applying-subschemas-conditionally
        
        VS OpenAPI Specification's Discriminated Unions
        https://swagger.io/docs/specification/data-models/inheritance-and-polymorphism/
        """
        allOf = []
        discriminator = element.get("discriminator")
        if not discriminator:
            raise ValueError("<choice /> elements must specify a discriminator!")
        discriminator_model = ModelSchema(
            type=ValidationType(SimpleTypes.STRING), enum=[]
        )
        for choice_case in element:
            case_name = choice_case.get("name")
            if not case_name:
                raise ValueError("<case /> elements must specify a name!")

            discriminator_model.enum.append(case_name)

            case_if_then_model = ModelSchema()
            case_if_then_properties = {}

            case_properties = {}
            required: List[str] = []
            for case_child in choice_case:
                case_child_name = case_child.get("name")
                child_required = case_child.get("required") == "true"
                if not case_child_name:
                    output_path = json_path.replace("$.", "output.")
                    logger.warn(
                        f"{output_path}.{case_name} has a nameless child"
                        " which is not allowed!"
                    )
                    continue
                if child_required:
                    required.append(case_child_name)
                case_child_schema = parse_element(
                    case_child, processed_schema, f"{json_path}.{case_child_name}"
                )
                case_properties[case_child_name] = case_child_schema.to_dict()

            case_if_then_properties[discriminator] = ModelSchema(
                const=case_name
            ).to_dict()
            case_if_then_model.var_if = ModelSchema(
                properties=case_if_then_properties
            ).to_dict()
            case_if_then_model.then = ModelSchema(
                properties=case_properties, required=required
            ).to_dict()
            allOf.append(case_if_then_model)

        properties = {}
        properties[discriminator] = discriminator_model.to_dict()
        return ModelSchema(
            type=ValidationType(SimpleTypes.OBJECT),
            properties=properties,
            required=[discriminator],
            allOf=allOf,
            description=description,
        )
    else:
        format = xml_to_string(element.attrib.get("format", schema_type))
        return ModelSchema(
            type=ValidationType(SimpleTypes.STRING),
            description=description,
            format=format,
        )


# def load_input(input: str, output_schema: Dict[str, Any]) -> str:
#     """Legacy behaviour to substitute constants in on init."""
#     const_subbed_input = substitute_constants(input)
#     return Template(const_subbed_input).safe_substitute(
#         output_schema=json.dumps(output_schema)
#     )


# def parse_input(
#     input_tag: _Element,
#     output_schema: Dict[str, Any],
#     processed_schema: ProcessedSchema,
#     meta_property: str,
# ) -> str:
#     parse_element(input_tag, processed_schema, json_path=meta_property)
#     # NOTE: Don't do this here.
#     # This used to happen during RAIL init,
#     #     but it's cleaner if we just keep it as a string.
#     # This way the Runner has a strict contract for inputs being strings
#     #   and it can format/process them however it needs to.
#     # input = load_input(input_tag.text, output_schema)
#     return input


def rail_string_to_schema(rail_string: str) -> ProcessedSchema:
    processed_schema = ProcessedSchema(
        validators=[], validator_map={}, exec_opts=GuardExecutionOptions()
    )

    XMLPARSER = XMLParser(encoding="utf-8")
    rail_xml: _Element = ET.fromstring(rail_string, parser=XMLPARSER)

    # Load <output /> schema
    output_element = rail_xml.find("output")
    if output_element is None:
        raise ValueError("RAIL must contain a output element!")

    # FIXME: Is this re-serialization & de-serialization necessary?
    utf8_output_element = ET.tostring(output_element, encoding="utf-8")
    marshalled_output_element = ET.fromstring(utf8_output_element, parser=XMLPARSER)

    output_schema = parse_element(marshalled_output_element, processed_schema)

    processed_schema.json_schema = output_schema.to_dict()

    if output_schema.type.actual_instance == SimpleTypes.STRING:
        processed_schema.output_type = OutputTypes.STRING
    elif output_schema.type.actual_instance == SimpleTypes.ARRAY:
        processed_schema.output_type = OutputTypes.LIST
    elif output_schema.type.actual_instance == SimpleTypes.OBJECT:
        processed_schema.output_type = OutputTypes.DICT
    else:
        raise ValueError(
            "The type attribute of the <output /> tag must be one of:"
            ' "string", "object", or "list"'
        )

    # Parse instructions for the LLM. These are optional but if given,
    # LLMs can use them to improve their output. Commonly these are
    # prepended to the prompt.
    instructions_tag = rail_xml.find("instructions")
    if instructions_tag is not None:
        processed_schema.exec_opts.instructions = parse_element(
            instructions_tag, processed_schema, "instructions"
        )

    # Load <prompt />
    prompt_tag = rail_xml.find("prompt")
    if prompt_tag is not None:
        processed_schema.exec_opts.prompt = parse_element(
            prompt_tag, processed_schema, "prompt"
        )

    # If reasking prompt and instructions are provided, add them to the schema.
    reask_prompt = rail_xml.find("reask_prompt")
    if reask_prompt is not None:
        processed_schema.exec_opts.reask_prompt = reask_prompt.text

    reask_instructions = rail_xml.find("reask_instructions")
    if reask_instructions is not None:
        processed_schema.exec_opts.reask_instructions = reask_instructions.text

    return processed_schema


def rail_file_to_schema(file_path: str) -> ProcessedSchema:
    with open(file_path, "r") as f:
        rail_xml = f.read()
    return rail_string_to_schema(rail_xml)
