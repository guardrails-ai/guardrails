import json
import regex
from typing import Dict, Optional, Tuple, Union

from guardrails.actions.reask import NonParseableReAsk
from guardrails.classes.output_type import OutputTypes
from guardrails.classes.validation.validation_result import FailResult


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


def extract_json_from_ouput(output: str) -> Tuple[Optional[Dict], Optional[Exception]]:
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


def parse_fragment(fragment: str):
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
        parsed_fragment = json.loads(fragment)
        return parsed_fragment, None
    except ValueError as e:
        return fragment, str(e)


### LLM Output Parsing ###
def parse_json_llm_output(
    output: str, **kwargs
) -> Tuple[
    Union[Optional[Dict], NonParseableReAsk, str],
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
