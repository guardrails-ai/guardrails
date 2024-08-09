import jsonref
import warnings
from dataclasses import dataclass
from string import Template
from typing import Any, Callable, Dict, List, Optional, Tuple, cast
from guardrails_api_client.models.validation_type import ValidationType
from lxml import etree as ET
from lxml.etree import _Element, Element, SubElement, XMLParser
from xml.etree.ElementTree import canonicalize
from guardrails_api_client import ModelSchema, SimpleTypes, ValidatorReference
from guardrails.classes.execution.guard_execution_options import GuardExecutionOptions
from guardrails.classes.output_type import OutputTypes
from guardrails.classes.schema.processed_schema import ProcessedSchema
from guardrails.logger import logger
from guardrails.types import RailTypes
from guardrails.types.validator import ValidatorMap
from guardrails.utils.regex_utils import split_on
from guardrails.utils.validator_utils import get_validator
from guardrails.utils.xml_utils import xml_to_string
from guardrails.validator_base import OnFailAction, Validator


### RAIL to JSON Schema ###
STRING_TAGS = [
    "instructions",
    "prompt",
    "reask_instructions",
    "reask_prompt",
    "messages",
    "reask_messages",
]


def parse_on_fail_handlers(element: _Element) -> Dict[str, OnFailAction]:
    on_fail_handlers: Dict[str, OnFailAction] = {}
    for key, value in element.attrib.items():
        key = xml_to_string(key) or ""
        if key.startswith("on-fail-"):
            on_fail_handler_name = key[len("on-fail-") :]
            on_fail_handler = OnFailAction(value)
            on_fail_handlers[on_fail_handler_name] = on_fail_handler
    return on_fail_handlers


def get_validators(element: _Element) -> List[Validator]:
    validators_string: str = xml_to_string(element.attrib.get("validators", "")) or ""
    validator_specs = split_on(validators_string, ";")
    on_fail_handlers = parse_on_fail_handlers(element)
    validators: List[Validator] = []
    for v in validator_specs:
        validator: Validator = get_validator(v)
        if not validator:
            continue
        on_fail = on_fail_handlers.get(
            validator.rail_alias.replace("/", "_"), OnFailAction.NOOP
        )
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
            on_fail=validator.on_fail_descriptor,  # type: ignore
            kwargs=validator.get_args(),
        )
        processed_schema.validators.append(validator_reference)

    if validators:
        path_validators = processed_schema.validator_map.get(json_path, [])
        path_validators.extend(validators)
        processed_schema.validator_map[json_path] = path_validators


def extract_format(
    element: _Element,
    internal_type: RailTypes,
    internal_format_attr: str,
) -> Optional[str]:
    """Prioritizes information retention over custom formats.

    Example:
        RAIL - <date format="foo" date-format="%Y-%M-%D" />
        JSON Schema - { "type": "string", "format": "date: %Y-%M-%D; foo" }
    """
    custom_format = ""
    format = xml_to_string(element.attrib.get("format", internal_type))
    if format != internal_type:
        custom_format = format
        format = internal_type
    internal_format = xml_to_string(element.attrib.get(internal_format_attr))
    if internal_format:
        format = Template("${format}: ${internal_format}").safe_substitute(
            format=format, internal_format=internal_format
        )
    if custom_format:
        format = Template("${format}; ${custom_format};").safe_substitute(
            format=format, custom_format=custom_format
        )
    return format


def parse_element(
    element: _Element, processed_schema: ProcessedSchema, json_path: str = "$"
) -> ModelSchema:
    """Takes an XML element Extracts validators to add to the 'validators' list
    and validator_map Returns a ModelSchema."""
    schema_type = element.tag
    if element.tag in STRING_TAGS:
        schema_type = RailTypes.STRING
    elif element.tag == "output":
        schema_type: str = element.attrib.get("type", RailTypes.OBJECT)  # type: ignore

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
        format = extract_format(
            element=element,
            internal_type=RailTypes.DATE,
            internal_format_attr="date-format",
        )
        return ModelSchema(
            type=ValidationType(SimpleTypes.STRING),
            description=description,
            format=format,
        )
    elif schema_type == RailTypes.TIME:
        format = extract_format(
            element=element,
            internal_type=RailTypes.TIME,
            internal_format_attr="time-format",
        )
        return ModelSchema(
            type=ValidationType(SimpleTypes.STRING),
            description=description,
            format=format,
        )
    elif schema_type == RailTypes.DATETIME:
        format = extract_format(
            element=element,
            internal_type=RailTypes.DATETIME,
            internal_format_attr="datetime-format",
        )
        return ModelSchema(
            type=ValidationType(SimpleTypes.STRING),
            description=description,
            format=format,
        )
    elif schema_type == RailTypes.PERCENTAGE:
        format = extract_format(
            element=element,
            internal_type=RailTypes.PERCENTAGE,
            internal_format_attr="",
        )
        return ModelSchema(
            type=ValidationType(SimpleTypes.STRING),
            description=description,
            format=format,
        )
    elif schema_type == RailTypes.ENUM:
        format = xml_to_string(element.attrib.get("format"))
        csv = xml_to_string(element.attrib.get("values", "")) or ""
        values = [v.strip() for v in csv.split(",")] if csv else None
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
        if num_of_children > 1:
            raise ValueError(
                "<list /> RAIL elements must have precisely 1 child element!"
            )
        elif num_of_children == 0:
            items = {}
        else:
            first_child = children[0]
            child_schema = parse_element(
                first_child, processed_schema, f"{json_path}.*"
            )
            items = child_schema.to_dict()
        return ModelSchema(
            type=ValidationType(SimpleTypes.ARRAY), items=items, description=description
        )
    elif schema_type == RailTypes.OBJECT:
        properties = {}
        required: List[str] = []
        for child in element:
            name = child.get("name")
            child_required = child.get("required", "true") == "true"
            if not name:
                output_path = json_path.replace("$.", "output.")
                logger.warning(
                    f"{output_path} has a nameless child which is not allowed!"
                )
                continue
            if child_required:
                required.append(name)
            child_schema = parse_element(child, processed_schema, f"{json_path}.{name}")
            properties[name] = child_schema.to_dict()

        object_schema = ModelSchema(
            type=ValidationType(SimpleTypes.OBJECT),
            properties=properties,
            description=description,
            required=required,
        )
        if not properties:
            object_schema.additional_properties = True
        return object_schema
    elif schema_type == RailTypes.CHOICE:
        """Since our ModelSchema class reflects the pure JSON Schema structure
        this implementation of choice-case strays from the Discriminated Unions
        specification as defined by OpenAPI that Pydantic uses.

        We should verify that LLM's understand this syntax properly. If
        they do not, we can manually add the 'discriminator' property to
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

            discriminator_model.enum.append(case_name)  # type: ignore

            case_if_then_model = ModelSchema()
            case_if_then_properties = {}

            case_properties = {}
            required: List[str] = []
            for case_child in choice_case:
                case_child_name = case_child.get("name")
                child_required = case_child.get("required", "true") == "true"
                if not case_child_name:
                    output_path = json_path.replace("$.", "output.")
                    logger.warning(
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
        # TODO: What if the user specifies a custom tag _and_ a format?
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

    XMLPARSER = XMLParser(encoding="utf-8", resolve_entities=False)
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

    output_schema_type = output_schema.type
    if not output_schema_type:
        raise ValueError(
            "The type attribute of the <output /> tag must be one of:"
            ' "string", "object", or "list"'
        )
    if output_schema_type.actual_instance == SimpleTypes.STRING:
        processed_schema.output_type = OutputTypes.STRING
    elif output_schema_type.actual_instance == SimpleTypes.ARRAY:
        processed_schema.output_type = OutputTypes.LIST
    elif output_schema_type.actual_instance == SimpleTypes.OBJECT:
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
        parse_element(instructions_tag, processed_schema, "instructions")
        processed_schema.exec_opts.instructions = instructions_tag.text
        warnings.warn(
            "The instructions tag has been deprecated"
            " in favor of messages. Please use messages instead.",
            DeprecationWarning,
        )

    # Load <prompt />
    prompt_tag = rail_xml.find("prompt")
    if prompt_tag is not None:
        parse_element(prompt_tag, processed_schema, "prompt")
        processed_schema.exec_opts.prompt = prompt_tag.text
        warnings.warn(
            "The prompt tag has been deprecated"
            " in favor of messages. Please use messages instead.",
            DeprecationWarning,
        )

    # If reasking prompt and instructions are provided, add them to the schema.
    reask_prompt = rail_xml.find("reask_prompt")
    if reask_prompt is not None:
        processed_schema.exec_opts.reask_prompt = reask_prompt.text
        warnings.warn(
            "The reask_prompt tag has been deprecated"
            " in favor of reask_messages. Please use reask_messages instead.",
            DeprecationWarning,
        )

    reask_instructions = rail_xml.find("reask_instructions")
    if reask_instructions is not None:
        processed_schema.exec_opts.reask_instructions = reask_instructions.text
        warnings.warn(
            "The reask_instructions tag has been deprecated"
            " in favor of reask_messages. Please use reask_messages instead.",
            DeprecationWarning,
        )

    messages = rail_xml.find("messages")
    if messages is not None:
        extracted_messages = []
        for msg in messages:
            if msg.tag == "message":
                message = msg
                role = message.attrib.get("role")
                content = message.text
                extracted_messages.append({"role": role, "content": content})
        processed_schema.exec_opts.messages = extracted_messages

    reask_messages = rail_xml.find("reask_messages")
    if reask_messages is not None:
        extracted_reask_messages = []
        for msg in reask_messages:
            if msg.tag == "message":
                message = msg
                role = message.attrib.get("role")
                content = message.text
                extracted_reask_messages.append({"role": role, "content": content})
        processed_schema.exec_opts.messages = extracted_reask_messages

    return processed_schema


def rail_file_to_schema(file_path: str) -> ProcessedSchema:
    with open(file_path, "r") as f:
        rail_xml = f.read()
    return rail_string_to_schema(rail_xml)


### JSON Schema to RAIL ###
@dataclass
class Format:
    internal_type: Optional[RailTypes] = None
    internal_format_attr: Optional[str] = None
    custom_format: Optional[str] = None

    def __repr__(self):
        return f"Format(internal_type={self.internal_type},internal_format_attr={self.internal_format_attr},custom_format={self.custom_format})"  # noqa


def extract_internal_format(format: str) -> Format:
    fmt = Format()

    internal, *custom_rest = format.split("; ")

    fmt.custom_format = "; ".join(custom_rest)

    internal_type, *format_attr_rest = internal.split(": ")

    if not RailTypes.get(internal_type):
        # This format wasn't manipulated by us,
        # it just happened to match our pattern
        fmt.custom_format = format
        return fmt

    fmt.internal_type = RailTypes.get(internal_type)
    fmt.internal_format_attr = ": ".join(format_attr_rest)

    return fmt


def init_elem(
    elem: Callable[..., _Element] = SubElement,
    *,
    _tag: str,
    attrib: Dict[str, Any],
    _parent: Optional[_Element] = None,
) -> _Element:
    if elem == Element:
        return Element(_tag, attrib)
    elif _parent is not None:
        return SubElement(_parent, _tag, attrib)
    # This should never happen unless we mess up the code.
    raise RuntimeError("rail_schema.py::init_elem() was called with no parent!")


def build_list_element(
    json_schema: Dict[str, Any],
    validator_map: ValidatorMap,
    attributes: Dict[str, Any],
    *,
    json_path: str = "$",
    elem: Callable[..., _Element] = SubElement,
    tag_override: Optional[str] = None,
    parent: Optional[_Element] = None,
) -> _Element:
    rail_type = RailTypes.LIST
    tag = tag_override or rail_type
    element = init_elem(elem, _parent=parent, _tag=tag, attrib=attributes)

    item_schema = json_schema.get("items")
    if item_schema:
        build_element(item_schema, validator_map, json_path=json_path, parent=element)
    return element


def build_choice_case(
    *,
    cases: List[Dict[str, Any]],
    attributes: Dict[str, str],
    parent: _Element,
    validator_map: ValidatorMap,
    json_path: str,
    discriminator: Optional[str] = None,
) -> _Element:
    choice_attributes = {**attributes}
    if discriminator:
        choice_attributes["discriminator"] = discriminator
    choice = SubElement(parent, RailTypes.CHOICE, choice_attributes)
    for case in cases:
        case_attributes = {}
        case_value = case.get("case")
        if case_value:
            case_attributes["name"] = case_value
        case_elem = SubElement(
            _parent=choice, _tag=RailTypes.CASE, attrib=case_attributes
        )

        case_schema: Dict[str, Any] = case.get("schema", {})
        case_properties: Dict[str, Any] = case_schema.get("properties", {})
        case_required_list: List[str] = case_schema.get("required", [])
        for ck, cv in case_properties.items():
            required = ck in case_required_list
            build_element(
                cv,
                validator_map,
                json_path=f"{json_path}.{ck}",
                parent=case_elem,
                required=str(required).lower(),
                attributes={"name": ck},
            )
    return choice


def build_choice_case_element_from_if(
    json_schema: Dict[str, Any],
    validator_map: ValidatorMap,
    attributes: Dict[str, Any],
    *,
    json_path: str = "$",
    elem: Callable[..., _Element] = SubElement,
    parent: Optional[_Element] = None,
) -> _Element:
    choice_name = json_path.split(".")[-1]
    attributes["name"] = choice_name

    properties: Dict[str, Any] = json_schema.get("properties", {})
    all_of: List[Dict[str, Any]] = json_schema.get("allOf", [])

    # Non-conditional inclusions
    other_subs: List[Dict[str, Any]] = [sub for sub in all_of if not sub.get("if")]
    factored_properties: Dict[str, Any] = {**properties}
    for sub in other_subs:
        factored_properties = {**factored_properties, **sub}

    # Conditional inclusions
    if_subs: List[Dict[str, Any]] = [sub for sub in all_of if sub.get("if")]

    # { discriminator: List[case] }
    discriminator_combos: Dict[str, List[Dict[str, Any]]] = {}

    for if_sub in if_subs:
        if_block: Dict[str, Any] = if_sub.get("if", {})
        then_block: Dict[str, Any] = if_sub.get("then", {})
        else_block: Dict[str, Any] = if_sub.get("else", {})

        if_props: Dict[str, Dict] = if_block.get("properties", {})
        discriminators: List[str] = []
        cases: List[str] = []
        for k, v in if_props.items():
            discriminators.append(k)
            case_value: str = v.get("const", "")
            cases.append(case_value)

        joint_discriminator = ",".join(discriminators)
        joint_case = ",".join(cases)
        case_combo = discriminator_combos.get(joint_discriminator, [])

        then_schema = {
            k: v
            for k, v in {**factored_properties, **then_block}.items()
            if k not in discriminators
        }
        case_combo.append({"case": joint_case, "schema": then_schema})

        if else_block:
            else_schema = {
                k: v
                for k, v in {**factored_properties, **else_block}.items()
                if k not in discriminators
            }
            case_combo.append(
                {"discriminator": joint_discriminator, "schema": else_schema}
            )
        discriminator_combos[joint_discriminator] = case_combo

    if len(discriminator_combos) > 1:
        # FIXME: This can probably be refactored
        anonymous_choice = init_elem(
            elem, _parent=parent, _tag=RailTypes.CHOICE, attrib={}
        )
        for discriminator, discriminator_cases in discriminator_combos.items():
            anonymous_case = SubElement(_parent=anonymous_choice, _tag=RailTypes.CASE)
            build_choice_case(
                discriminator=discriminator,
                cases=discriminator_cases,
                attributes=attributes,
                parent=anonymous_case,
                validator_map=validator_map,
                json_path=json_path,
            )
        return anonymous_choice
    else:
        first_discriminator: Tuple[str, List[Dict[str, Any]]] = list(
            discriminator_combos.items()
        )[0] or ("", [])
        discriminator, discriminator_cases = first_discriminator
        return build_choice_case(
            discriminator=discriminator,
            cases=discriminator_cases,
            attributes=attributes,
            parent=parent,  # type: ignore
            validator_map=validator_map,
            json_path=json_path,
        )


def build_choice_case_element_from_discriminator(
    json_schema: Dict[str, Any],
    validator_map: ValidatorMap,
    attributes: Dict[str, Any],
    *,
    json_path: str = "$",
    parent: Optional[_Element] = None,
) -> _Element:
    """Takes an OpenAPI Spec flavored JSON Schema with a discriminated union.

    Returns a choice-case RAIL element.
    """
    one_of: List[Dict[str, Any]] = json_schema.get("oneOf", [])
    discriminator_container: Dict[str, Any] = json_schema.get("discriminator", {})
    discriminator = discriminator_container.get("propertyName")
    discriminator_map: Dict[str, Any] = discriminator_container.get("mapping", {})
    case_values = discriminator_map.keys()

    cases = []
    for sub in one_of:
        sub_schema = {
            **sub,
            "properties": {
                k: v for k, v in sub.get("properties", {}).items() if k != discriminator
            },
        }
        case = {"schema": sub_schema}
        discriminator_value = (
            sub.get("properties", {}).get(discriminator, {}).get("const")
        )
        if discriminator_value in case_values:
            case["case"] = discriminator_value
        cases.append(case)

    return build_choice_case(
        cases=cases,
        attributes=attributes,
        parent=parent,  # type: ignore
        validator_map=validator_map,
        json_path=json_path,
        discriminator=discriminator,
    )


def build_object_element(
    json_schema: Dict[str, Any],
    validator_map: ValidatorMap,
    attributes: Dict[str, Any],
    *,
    json_path: str = "$",
    elem: Callable[..., _Element] = SubElement,
    tag_override: Optional[str] = None,
    parent: Optional[_Element] = None,
) -> _Element:
    properties: Dict[str, Any] = json_schema.get("properties", {})

    # We don't entertain the possibility of using
    # multiple schema compositions in the same sub-schema.
    # Technically you _can_, but that doesn't mean you should.
    all_of: List[Dict[str, Any]] = json_schema.get("allOf", [])

    one_of = json_schema.get("oneOf", [])

    any_of = [
        sub
        for sub in json_schema.get("anyOf", [])
        if sub.get("type") != SimpleTypes.NULL
    ]

    all_of_contains_if = [sub for sub in all_of if sub.get("if")]
    discriminator = json_schema.get("discriminator")
    if all_of and all_of_contains_if:
        return build_choice_case_element_from_if(
            json_schema,
            validator_map,
            attributes,
            json_path=json_path,
            elem=elem,
            parent=parent,
        )
    elif one_of and discriminator:
        return build_choice_case_element_from_discriminator(
            json_schema, validator_map, attributes, json_path=json_path, parent=parent
        )
    elif all_of:
        factored_properties = {**properties}
        for sub in all_of:
            factored_properties = {**factored_properties, **sub}
        factored_schema = {**json_schema, "properties": factored_properties}
        factored_schema.pop("allOf", [])
        return build_object_element(
            json_schema,
            validator_map,
            attributes,
            json_path=json_path,
            elem=elem,
            tag_override=tag_override,
            parent=parent,
        )
    elif any_of or one_of:
        sub_schemas = any_of or one_of

        if len(sub_schemas) == 1:
            sub_schema = sub_schemas[0]
            factored_schema = {**json_schema, **sub_schema}
            factored_schema.pop("anyOf", [])
            factored_schema.pop("oneOf", [])
            return build_element(
                json_schema=sub_schema,
                validator_map=validator_map,
                json_path=json_path,
                elem=elem,
                tag_override=tag_override,
                parent=parent,
                attributes=attributes,
                required=attributes.get("required"),
            )
        else:
            cases = [{"schema": sub} for sub in sub_schemas]

            return build_choice_case(
                cases=cases,
                attributes=attributes,
                parent=parent,  # type: ignore
                validator_map=validator_map,
                json_path=json_path,
            )

    rail_type = RailTypes.OBJECT
    tag = tag_override or rail_type
    element = init_elem(elem, _parent=parent, _tag=tag, attrib=attributes)
    required_list = json_schema.get("required", [])
    for k, v in properties.items():
        child_path = f"{json_path}.{k}"
        required = k in required_list
        required_attr = str(required).lower()
        build_element(
            v,
            validator_map,
            json_path=child_path,
            parent=element,
            required=required_attr,
            attributes={"name": k},
        )
    return element


def build_string_element(
    json_schema: Dict[str, Any],
    attributes: Dict[str, Any],
    format: Format,
    *,
    elem: Callable[..., _Element] = SubElement,
    tag_override: Optional[str] = None,
    parent: Optional[_Element] = None,
) -> _Element:
    enum_values: List[str] = json_schema.get("enum", [])
    if enum_values:
        attributes["values"] = ", ".join(enum_values)
        tag = tag_override or RailTypes.ENUM
        if tag_override:
            attributes["type"] = RailTypes.ENUM
        return init_elem(elem, _parent=parent, _tag=tag, attrib=attributes)

    # Exit early if we can
    if not format.internal_type:
        tag = tag_override or RailTypes.STRING
        if tag_override:
            attributes["type"] = RailTypes.STRING
        return init_elem(elem, _parent=parent, _tag=tag, attrib=attributes)

    tag = tag_override or RailTypes.STRING
    type = RailTypes.STRING
    if format.internal_type == RailTypes.DATE:
        type = RailTypes.DATE
        tag = tag_override or RailTypes.DATE
        date_format = format.internal_format_attr
        if date_format:
            attributes["date-format"] = date_format
    elif format.internal_type == RailTypes.TIME:
        type = RailTypes.TIME
        tag = tag_override or RailTypes.TIME
        time_format = format.internal_format_attr
        if time_format:
            attributes["time-format"] = time_format
    elif format.internal_type == RailTypes.DATETIME:
        type = RailTypes.DATETIME
        tag = tag_override or RailTypes.DATETIME
        datetime_format = format.internal_format_attr
        if datetime_format:
            attributes["datetime-format"] = datetime_format
    elif format.internal_type == RailTypes.PERCENTAGE:
        type = RailTypes.PERCENTAGE
        tag = tag_override or RailTypes.PERCENTAGE

    if tag_override:
        attributes["type"] = type
    return init_elem(elem, _parent=parent, _tag=tag, attrib=attributes)


def build_element(
    json_schema: Dict[str, Any],
    validator_map: ValidatorMap,
    *,
    json_path: str = "$",
    elem: Callable[..., _Element] = SubElement,
    tag_override: Optional[str] = None,
    parent: Optional[_Element] = None,
    required: Optional[str] = "true",
    attributes: Optional[Dict[str, Any]] = None,
) -> _Element:
    """Takes an XML element Extracts validators to add to the 'validators' list
    and validator_map Returns a ModelSchema."""
    attributes = attributes or {}
    schema_type = json_schema.get("type", "object")

    description = json_schema.get("description")
    if description:
        attributes["description"] = description

    if required:
        attributes["required"] = required
    if tag_override:
        attributes.pop("required", "")

    format: Format = extract_internal_format(json_schema.get("format", ""))

    validators: List[Validator] = []
    validators.extend(validator_map.get(json_path, []))
    validators.extend(validator_map.get(f"{json_path}.*", []))

    # While we now require validators to be specified in rail
    #   using the 'validators' attribute,
    # Schema2Prompt still assigned these to 'format' for prompting
    rail_format: List[str] = [v.to_prompt(False) for v in validators]
    if format.custom_format:
        rail_format.insert(0, format.custom_format)
    rail_format_str = "; ".join(rail_format)
    if rail_format_str:
        attributes["format"] = rail_format_str

    rail_type = None
    if schema_type == SimpleTypes.ARRAY:
        return build_list_element(
            json_schema,
            validator_map,
            attributes,
            json_path=json_path,
            elem=elem,
            tag_override=tag_override,
            parent=parent,
        )
    elif schema_type == SimpleTypes.BOOLEAN:
        rail_type = RailTypes.BOOL
    elif schema_type == SimpleTypes.INTEGER:
        rail_type = RailTypes.INTEGER
    elif schema_type == SimpleTypes.NUMBER:
        # Special Case for Doc Examples
        if format.internal_type == RailTypes.PERCENTAGE:
            rail_format_str = "; ".join([RailTypes.PERCENTAGE, *rail_format])
            attributes["format"] = rail_format_str
        rail_type = RailTypes.FLOAT
    elif schema_type == SimpleTypes.OBJECT:
        """Checks for objects and choice-case."""
        return build_object_element(
            json_schema,
            validator_map,
            attributes,
            json_path=json_path,
            elem=elem,
            tag_override=tag_override,
            parent=parent,
        )
    elif schema_type == SimpleTypes.STRING:
        """Checks for string, date, time, datetime, enum."""
        return build_string_element(
            json_schema,
            attributes,
            format,
            elem=elem,
            tag_override=tag_override,
            parent=parent,
        )
    # This isn't possible in RAIL
    # elif schema_type == SimpleTypes.NULL:
    else:
        rail_type = RailTypes.STRING

    # Fall through logic for non-special cases
    tag = tag_override or rail_type
    element = init_elem(elem, _parent=parent, _tag=tag, attrib=attributes)
    return element


def json_schema_to_rail_output(
    json_schema: Dict[str, Any], validator_map: ValidatorMap
) -> str:
    """Takes a JSON Schema and converts it to the RAIL output specification.

    Limited support. Only guaranteed to work for JSON Schemas that were
    derived from RAIL.
    """
    dereferenced_json_schema = cast(Dict[str, Any], jsonref.replace_refs(json_schema))
    output_element = build_element(
        dereferenced_json_schema,
        validator_map,
        json_path="$",
        elem=Element,
        tag_override="output",
    )
    return canonicalize(ET.tostring(output_element, pretty_print=True)).replace(
        "&#xA;", ""
    )
