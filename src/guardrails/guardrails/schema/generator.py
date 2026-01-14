import jsonref
import re
import rstr
from builtins import max as get_max
from typing import Any, Dict, List, Optional, Union, cast
from pydash import upper_first, snake_case, camel_case, start_case, uniq_with, is_equal
from faker import Faker
from random import randint, randrange, uniform
from guardrails_api_client import SimpleTypes
from guardrails.utils.safe_get import safe_get

fake = Faker()


def get_decimal_places(num: Union[int, float]) -> int:
    return len(safe_get(str(num).split("."), 1, ""))


def closest_multiple(n, x):
    if x > n:
        return x
    z = (int)(x / 2)
    n = n + z
    n = n - (n % x)
    return n


def is_number(value: Any) -> bool:
    return str(value).replace(".", "").isnumeric()


def gen_sentence_case():
    words = " ".join(fake.words(2))
    return upper_first(words)


def gen_snake_case():
    words = " ".join(fake.words(2))
    return snake_case(words)


def gen_camel_case():
    words = " ".join(fake.words(2))
    return camel_case(words)


def gen_title_case():
    words = " ".join(fake.words(2))
    return start_case(words)


def gen_num(schema: Dict[str, Any]) -> Union[int, float]:
    schema_type = schema.get("type")
    minimum = schema.get("minimum")
    exclusive_minimum = schema.get("exclusiveMinimum")
    maximum = schema.get("maximum")
    exclusive_maximum = schema.get("exclusiveMaximum")
    multiple_of = schema.get("multipleOf")

    num_type = int if schema_type == SimpleTypes.INTEGER else float

    step = 1
    if multiple_of and is_number(multiple_of):
        step = multiple_of
    elif schema_type != SimpleTypes.INTEGER:
        specified_min = minimum or exclusive_minimum or 0
        specified_max = maximum or exclusive_maximum or 100
        min_digits = get_decimal_places(specified_min)
        max_digits = get_decimal_places(specified_max)
        highest_precision = get_max(min_digits, max_digits)
        step = float(f"{0:.{highest_precision - 1}f}1") if highest_precision else 1

    min = 0
    max = 100
    if minimum and str(minimum).isnumeric():
        min = minimum
    elif exclusive_minimum and is_number(exclusive_minimum):
        min = num_type(exclusive_minimum) + step

    if maximum and is_number(maximum):
        max = maximum
    elif exclusive_maximum and is_number(exclusive_maximum):
        max = num_type(exclusive_maximum) - step

    random_num = 0
    if schema_type == SimpleTypes.INTEGER or isinstance(step, int):
        random_num = num_type(randrange(min, max, step))  # type: ignore
    else:
        precision = get_decimal_places(step)
        random_num = round(num_type(uniform(min, max)), precision)
        random_num = round(closest_multiple(random_num, step), precision)

    return random_num


def gen_formatted_string(format: str, default: str) -> str:
    value = default
    if format == "date":
        value = fake.date("YYYY-MM-DD")
    elif format == "date-time":
        value = fake.date_time_this_century().isoformat("T")
    elif format == "time":
        value = fake.time()
    elif format == "percentage":
        value = f"{round(uniform(0, 100), 2)}%"
    elif format == "email":
        value = fake.email()
    elif format == "url" or format == "uri":
        value = fake.url()
    elif format == "snake_case":
        value = gen_snake_case()
    elif format == "regex":
        value = ".*"
    elif format == "camelCase":
        value = gen_camel_case()
    elif format == "Title Case":
        value = gen_title_case()
    elif hasattr(fake, format) and callable(getattr(fake, format)):
        gen_func = getattr(fake, format)
        value = gen_func()

    return value


def gen_string(schema: Dict[str, Any], *, property_name: Optional[str] = None) -> str:
    # Look at format first, then pattern; not xor
    gen_func = fake.word
    # Lazy attempt to choose a relevant faker function
    if (
        property_name
        and hasattr(fake, property_name)
        and callable(getattr(fake, property_name))
    ):
        gen_func = getattr(fake, property_name)

    value = gen_func()

    schema_format = schema.get("format")
    if schema_format:
        value = gen_formatted_string(schema_format, value)

    schema_pattern = schema.get("pattern")
    regex_pattern = re.compile(schema_pattern) if schema_pattern else None  # type: ignore
    if schema_pattern and regex_pattern and not regex_pattern.search(value):
        value = rstr.xeger(schema_pattern)

    return value


def gen_array(
    schema: Dict[str, Any], *, property_name: Optional[str] = None
) -> List[Any]:
    """
    What we do support:
        - items
        - minItems
        - maxItem
        - uniqueItems
    What we do NOT support:
        - prefixItems
        - unevaluatedItems
        - contains
    """
    item_schema = schema.get("items", {})
    min_items = schema.get("minItems", 1)
    max_item = schema.get("maxItem", 2)
    unique_items = schema.get("uniqueItems", False)

    gen_amount = randint(min_items, max_item)

    array_items = []
    while len(array_items) < gen_amount:
        item = _generate_example(item_schema, property_name=property_name)
        array_items.append(item)
        if unique_items:
            array_items = uniq_with(array_items, is_equal)

    return array_items


def gen_object(schema: Dict[str, Any]) -> Dict[str, Any]:
    """
    What we do support:
        - properties
        - schema compositions: Addressed in _generate_example
            - oneOf
            - anyOf
            - allOf
        - conditional sub-schemas: Addressed in _generate_example
            - if/then/else
            - allOf[if/then/else]

    What we do NOT support:
            - patternProperties
            - additionalProperties
            - unevaluatedProperties
            - propertyNames
            - minProperties
            - maxProperties
            - dependentSchemas (just use anyOf)
            - dependentRequired (we generate all properties; so this is validation only)
    """
    value = {}
    properties: Dict[str, Any] = schema.get("properties", {})
    for k, v in properties.items():
        value[k] = _generate_example(v, property_name=k)

    return value


def gen_from_type(
    schema: Dict[str, Any], *, property_name: Optional[str] = None
) -> Any:
    schema_type = schema.get("type")
    if schema_type == SimpleTypes.ARRAY:
        return gen_array(schema, property_name=property_name)
    elif schema_type == SimpleTypes.BOOLEAN:
        return fake.boolean()
    elif schema_type == SimpleTypes.INTEGER:
        return gen_num(schema)
    elif schema_type == SimpleTypes.NULL:
        return None
    elif schema_type == SimpleTypes.NUMBER:
        return gen_num(schema)
    elif schema_type == SimpleTypes.OBJECT:
        return gen_object(schema)
    elif schema_type == SimpleTypes.STRING:
        return gen_string(schema, property_name=property_name)


def gen_from_enum(enum: List[Any]) -> Any:
    random_enum_index = randint(0, len(enum) - 1)
    return safe_get(enum, random_enum_index)


def evaluate_if_block(schema: Dict[str, Any], value: Any) -> Any:
    if_block: Dict[str, Any] = schema.get("if", {})
    if_properties: Dict[str, Any] = if_block.get("properties", {})

    then_block: Dict[str, Any] = schema.get("then", {})
    then_properties: Dict[str, Any] = then_block.get("properties", {})

    else_block: Dict[str, Any] = schema.get("else", {})
    else_properties: Dict[str, Any] = else_block.get("properties", {})

    sub_schema = else_properties

    condition_satisfied = True
    for k, v in if_properties.items():
        actual_value = safe_get(value, k)
        condition_value = safe_get(v, "const")
        condition_satisfied = condition_satisfied and actual_value == condition_value

    if condition_satisfied:
        sub_schema = then_properties

    for k, v in sub_schema.items():
        sub_schema_value = _generate_example(v, property_name=k)
        value[k] = sub_schema_value

    return value


def pick_sub_schema(
    schema: Dict[str, Any], sub_schema_key: str, *, property_name: Optional[str] = None
) -> Any:
    sub_schema: List[Dict[str, Any]] = schema.pop(sub_schema_key, [])
    # Pick a sub-schema
    random_index = randint(0, len(sub_schema) - 1)
    chosen_sub_schema = safe_get(sub_schema, random_index, {})

    # Factor
    factored_schema = {**schema, **chosen_sub_schema}
    return _generate_example(factored_schema, property_name=property_name)


def evaluate_all_of(
    schema: Dict[str, Any], value: Any, *, property_name: Optional[str] = None
) -> Any:
    # If 'type' isn't specified but 'allOf' is applied;
    #   it is safe to assume the schema is of type 'object'
    all_of: List[Dict[str, Any]] = schema.pop("allOf", [])
    schema_type = schema.get("type", SimpleTypes.OBJECT)
    if schema_type == SimpleTypes.OBJECT:
        # Check for "if" blocks, group by properties, pick one of each group
        # "if" blocks can _only_ be applied to objects
        if_blocks = [sub for sub in all_of if sub.get("if")]
        for if_block in if_blocks:
            factored_schema = {**schema, **if_block}
            value = evaluate_if_block(factored_schema, value)

        other_blocks = [sub for sub in all_of if not sub.get("if")]
        for sub_schema in other_blocks:
            sub_schema_value = _generate_example(
                sub_schema, property_name=property_name
            )
            value = {**value, **sub_schema_value}
        return value
    else:
        compressed_schema = {**schema}
        for sub_schema in all_of:
            compressed_schema.update(sub_schema)
        return _generate_example(compressed_schema, property_name=property_name)


def _generate_example(
    json_schema: Dict[str, Any], *, property_name: Optional[str] = None
) -> Any:
    # Apply base schema
    schema_type = json_schema.get("type")
    const = json_schema.get("const")
    enum = json_schema.get("enum")

    value = None
    if const:
        value = const
    elif enum:
        value = gen_from_enum(enum)
    elif schema_type:
        value = gen_from_type(json_schema, property_name=property_name)

    # Apply Conditional Schema
    if_block: Dict[str, Any] = json_schema.get("if", {})
    if if_block:
        value = evaluate_if_block(json_schema, value)
    # elif discriminator:
    #     # Don't need to evaluate this;
    #     # It is implied in the oneOf
    #     pass

    # Apply Schema Compositions
    one_of: List[Dict[str, Any]] = json_schema.get("oneOf", [])
    any_of: List[Dict[str, Any]] = json_schema.get("anyOf", [])
    all_of: List[Dict[str, Any]] = json_schema.get("allOf", [])
    if one_of:
        value = pick_sub_schema(json_schema, "oneOf")
    elif any_of:
        value = pick_sub_schema(json_schema, "anyOf")
    elif all_of:
        value = evaluate_all_of(json_schema, value, property_name=property_name)

    return value


def generate_example(
    json_schema: Dict[str, Any], *, property_name: Optional[str] = None
) -> Any:
    """Takes a json schema and generates a sample object."""
    dereferenced_schema = cast(Dict[str, Any], jsonref.replace_refs(json_schema))
    return _generate_example(dereferenced_schema, property_name=property_name)
