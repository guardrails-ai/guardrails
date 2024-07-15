import json
from guardrails_api_client import SimpleTypes
import jsonref
import regex
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union, cast

from guardrails.actions.reask import NonParseableReAsk
from guardrails.classes.output_type import OutputTypes
from guardrails.classes.validation.validation_result import FailResult
from guardrails.schema.parser import get_all_paths
from guardrails.utils.safe_get import safe_get


### String to Dictionary Parsing ###
def has_code_block(
    string_value: str, code_type: str = ""
) -> Tuple[bool, Optional[int], Optional[int]]:
    """Checks if a string contains a code block denoted by leading and trailing
    tripple ticks (```) with an optional code type for the opening tag.

    Args::
        string_value (str): The string to check.
        code_type (str, optional): The specific code type to check for. Defaults to empty string.

    Returns
        bool: Whether or not the string contains the specified type of code block.
        int: The starting index of the code block.
        int: The ending index of the code block.
    """  # noqa
    block_border = "```"
    block_start_border = f"{block_border}{code_type}"

    block_start_index = string_value.find(block_start_border)
    if block_start_index != -1:
        block_end_index = string_value.find(
            block_border, (block_start_index + len(block_start_border))
        )
        return (
            (True, block_start_index, block_end_index)
            if block_end_index != -1
            else (False, None, None)
        )
    return (False, None, None)


def get_code_block(
    string_value: str, start: int, end: int, code_type: Optional[str] = ""
) -> str:
    """Takes a string, start and end indexes, and an optional code type to
    extract a code block from a string.

    Args::
        string_value (str): The string to extract a the code block from.
        start (int): The starting index of the code block.  This is assumed to be inclusive of the block boundaries.
        end (int): The ending index of the code block.  This is assumed to be inclusive of the block boundaries.
        code_type (str, optional): The specific code type to check for. Defaults to empty string.

    Returns:
        str: The contents of the code block.
    """  # noqa
    trimmed_input = string_value

    block_border = "```"
    block_start_border = f"{block_border}{code_type}"

    start_index = start + len(block_start_border)

    contents = trimmed_input[start_index:end]

    trimmed_output = contents.strip()

    return trimmed_output


def extract_json_from_ouput(
    output: str,
) -> Tuple[Optional[Union[Dict, List]], Optional[Exception]]:
    # Find and extract json from code blocks
    extracted_code_block = output
    has_json_block, json_start, json_end = has_code_block(output, "json")
    if has_json_block and json_start is not None and json_end is not None:
        extracted_code_block = get_code_block(output, json_start, json_end, "json")
    else:
        has_block, block_start, block_end = has_code_block(output)
        if has_block and block_start is not None and block_end is not None:
            extracted_code_block = get_code_block(output, block_start, block_end)
        else:
            json_pattern = regex.compile(r"\{(?:[^{}]+|\{(?:(?R)|[^{}]+)*\})*\}")
            json_groups = json_pattern.findall(output)
            json_start, json_end = output.find("{"), output.rfind("}")
            if len(json_groups) > 0 and len(json_groups[0]) == (
                json_end - json_start + 1
            ):
                extracted_code_block = json_groups[0]

    # Treat the output as a JSON string, and load it into a dict.
    error = None
    try:
        output_as_dict = json.loads(extracted_code_block, strict=False)
    except json.decoder.JSONDecodeError as e:
        output_as_dict = None
        error = e
    return output_as_dict, error


### Streaming Fragment Parsing ###
def is_valid_fragment(fragment: str, verified: set) -> bool:
    """Check if the fragment is a somewhat valid JSON."""

    # Strip fragment of whitespaces and newlines
    # to avoid duplicate checks
    text = fragment.strip(" \n")

    # Check if text is already verified
    if text in verified:
        return False

    # Check if text is valid JSON
    try:
        json.loads(text)
        verified.add(text)
        return True
    except ValueError as e:
        error_msg = str(e)
        # Check if error is due to missing comma
        if "Expecting ',' delimiter" in error_msg:
            verified.add(text)
            return True
        return False


def parse_fragment(fragment: str) -> Tuple[Union[str, List, Dict], Optional[str]]:
    """Parse the fragment into a dict."""

    # Complete the JSON fragment to handle missing brackets
    # Stack to keep track of opening brackets
    stack = []

    # Process each character in the string
    for char in fragment:
        if char in "{[":
            # Push opening brackets onto the stack
            stack.append(char)
        elif char in "}]":
            # Pop from stack if matching opening bracket is found
            if stack and (
                (char == "}" and stack[-1] == "{") or (char == "]" and stack[-1] == "[")
            ):
                stack.pop()

    # Add the necessary closing brackets in reverse order
    while stack:
        opening_bracket = stack.pop()
        if opening_bracket == "{":
            fragment += "}"
        elif opening_bracket == "[":
            fragment += "]"

    # Parse the fragment
    try:
        parsed_fragment: Union[Dict, List] = json.loads(fragment)
        return parsed_fragment, None
    except ValueError as e:
        return fragment, str(e)


### LLM Output Parsing ###
def parse_json_llm_output(
    output: str, **kwargs
) -> Tuple[
    Union[str, List, Dict, NonParseableReAsk, None],
    Union[Optional[Exception], str, bool, None],
]:
    if kwargs.get("stream", False):
        # Do expected behavior for StreamRunner
        # 1. Check if the fragment is valid JSON
        verified = kwargs.get("verified", set())
        fragment_is_valid = is_valid_fragment(output, verified)
        if not fragment_is_valid:
            return output, True

        # 2. Parse the fragment
        parsed_fragment, parsing_error = parse_fragment(output)
        return parsed_fragment, parsing_error

    # Else do expected behavior for Runner
    # Try to get json code block from output.
    # Return error and reask if it is not parseable.
    parsed_output, error = extract_json_from_ouput(output)

    if error:
        reask = NonParseableReAsk(
            incorrect_value=output,
            fail_results=[
                FailResult(
                    fix_value=None,
                    error_message="Output is not parseable as JSON",
                )
            ],
        )
        return reask, error
    return parsed_output, None


def parse_string_llm_output(output: str) -> Tuple[str, Optional[Exception]]:
    # Return a ValueError if the output is empty, else None
    error = ValueError("Empty response received.") if not output else None
    return output, error


def parse_llm_output(output: str, output_type: OutputTypes, **kwargs):
    if output_type == OutputTypes.STRING:
        return parse_string_llm_output(output)
    return parse_json_llm_output(output, **kwargs)


def prune_extra_keys(
    payload: Union[str, List[Any], Dict[str, Any]],
    schema: Dict[str, Any],
    *,
    json_path: str = "$",
    all_json_paths: Optional[Set[str]] = None,
) -> Union[str, List[Any], Dict[str, Any]]:
    if all_json_paths is None or not len(all_json_paths):
        all_json_paths = get_all_paths(schema)

    if isinstance(payload, dict):
        # Do full lookbehind
        wildcards: List[str] = [
            path.split(".*")[0] for path in all_json_paths if ".*" in path
        ]
        ancestor_is_wildcard = any(w in json_path for w in wildcards)
        actual_keys = list(payload.keys())
        for key in actual_keys:
            child_path = f"{json_path}.{key}"
            if child_path not in all_json_paths and not ancestor_is_wildcard:
                del payload[key]
            else:
                prune_extra_keys(
                    payload=payload.get(key),  # type: ignore
                    schema=schema,
                    json_path=child_path,
                    all_json_paths=all_json_paths,
                )
    elif isinstance(payload, list):
        for item in payload:
            prune_extra_keys(
                payload=item,
                schema=schema,
                json_path=json_path,
                all_json_paths=all_json_paths,
            )

    return payload


def coerce(value: Any, desired_type: Callable) -> Any:
    try:
        coerced_value = desired_type(value)
        return coerced_value
    except (ValueError, TypeError):
        return value


def try_json_parse(value: str) -> Any:
    try:
        return json.loads(value)
    except Exception:
        return value


def coerce_to_type(
    payload: Union[str, List[Any], Dict[str, Any], Any], schema_type: SimpleTypes
) -> Any:
    if schema_type == SimpleTypes.ARRAY:
        if isinstance(payload, str):
            payload = try_json_parse(payload)
        if not isinstance(payload, list):
            return coerce(payload, list)
        return payload
    elif schema_type == SimpleTypes.BOOLEAN:
        if not isinstance(payload, bool):
            return coerce(payload, bool)
        return payload
    elif schema_type == SimpleTypes.INTEGER:
        if not isinstance(payload, int):
            val = coerce(payload, int)
            return val
        return payload
    elif schema_type == SimpleTypes.NULL:
        return None
    elif schema_type == SimpleTypes.NUMBER:
        if not isinstance(payload, float):
            return coerce(payload, float)
        return payload
    elif schema_type == SimpleTypes.OBJECT:
        if isinstance(payload, str):
            payload = try_json_parse(payload)
        if not isinstance(payload, dict):
            return coerce(payload, dict)
        return payload
    elif schema_type == SimpleTypes.STRING:
        if not isinstance(payload, str) and not isinstance(payload, (list, dict)):
            return coerce(payload, str)
        return payload


def coerce_property(
    payload: Union[str, List[Any], Dict[str, Any], Any], schema: Dict[str, Any]
) -> Union[str, List[Any], Dict[str, Any]]:
    schema_type = schema.get("type")
    if schema_type:
        payload = coerce_to_type(payload, schema_type)

    ### Schema Composition ###
    one_of = schema.get("oneOf")
    if one_of:
        possible_values = []
        for sub_schema in one_of:
            possible_values.append(coerce_property(payload, sub_schema))
            payload = safe_get(list(filter(None, possible_values)), 0, payload)

    any_of = schema.get("anyOf")
    if any_of:
        possible_values = []
        for sub_schema in any_of:
            possible_values.append(coerce_property(payload, sub_schema))
            payload = safe_get(list(filter(None, possible_values)), 0, payload)

    all_of: List[Dict[str, Any]] = schema.get("allOf", [])
    if all_of:
        if_blocks = [sub for sub in all_of if sub.get("if")]
        if if_blocks:
            for if_block in if_blocks:
                factored_schema = {**schema, **if_block}
                factored_schema.pop("allOf", {})
                payload = coerce_property(payload, factored_schema)

            other_blocks = [sub for sub in all_of if not sub.get("if")]
            for sub_schema in other_blocks:
                payload = coerce_property(payload, sub_schema)

        else:
            factored_schema = {**schema}
            factored_schema.pop("allOf")
            for sub_schema in all_of:
                factored_schema = {**schema, **sub_schema}
            payload = coerce_property(payload, factored_schema)

    ### Object Schema ###
    properties: Dict[str, Any] = schema.get("properties", {})
    if properties and isinstance(payload, dict):
        for k, v in properties.items():
            payload_value = payload.get(k)
            if payload_value:
                payload[k] = coerce_property(payload_value, v)

    ### Object Additional Properties ###
    additional_properties_schema: Dict[str, Any] = schema.get(
        "additionalProperties", {}
    )
    if isinstance(additional_properties_schema, bool):
        additional_properties_schema = {}
    if additional_properties_schema and isinstance(payload, dict):
        declared_properties = properties.keys()
        additional_properties = [
            key for key in payload.keys() if key not in declared_properties
        ]
        for prop in additional_properties:
            payload_value = payload.get(prop)
            if payload_value:
                payload[prop] = coerce_property(
                    payload_value, additional_properties_schema
                )

    ### Conditional SubSchema ###
    if_block: Dict[str, Any] = schema.get("if", {})
    if if_block and isinstance(payload, dict):
        if_properties: Dict[str, Any] = if_block.get("properties", {})

        then_block: Dict[str, Any] = schema.get("then", {})
        then_properties: Dict[str, Any] = then_block.get("properties", {})

        else_block: Dict[str, Any] = schema.get("else", {})
        else_properties: Dict[str, Any] = else_block.get("properties", {})

        conditional_schema = else_properties

        condition_satisfied = True
        for k, v in if_properties.items():
            actual_value = safe_get(payload, k)
            condition_value = safe_get(v, "const")
            condition_satisfied = (
                condition_satisfied and actual_value == condition_value
            )

        if condition_satisfied:
            conditional_schema = then_properties

        factored_schema = {**schema, "properties": {**properties, **conditional_schema}}
        factored_schema.pop("if", {})
        factored_schema.pop("then", {})
        factored_schema.pop("else", {})
        payload = coerce_property(payload, factored_schema)

    ### Array Schema ###
    item_schema: Dict[str, Any] = schema.get("items", {})
    if isinstance(payload, list) and item_schema:
        coerced_items = []
        for item in payload:
            coerced_items.append(coerce_property(item, item_schema))
        payload = coerced_items

    return payload


def coerce_types(
    payload: Union[str, List[Any], Dict[str, Any], Any], schema: Dict[str, Any]
) -> Union[str, List[Any], Dict[str, Any]]:
    dereferenced_schema = cast(
        Dict[str, Any], jsonref.replace_refs(schema)
    )  # for pyright
    return coerce_property(payload, dereferenced_schema)
