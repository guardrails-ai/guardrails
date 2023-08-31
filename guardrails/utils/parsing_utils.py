from typing import Optional


def has_code_block(
    string_value: str, code_type: str = ""
) -> (bool, Optional[int], Optional[int]):
    """Checks if a string contains a code block denoted by leading and trailing
    tripple ticks (```) with an optional code type for the opening tag.

    Parameters: Arguments:
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
            else (False)
        )
    return (False, None, None)


def get_code_block(
    string_value: str, start: int, end: int, code_type: Optional[str] = ""
) -> str:
    """Takes a string, start and end indexes, and an optional code type to
    extract a code block from a string.

    Parameters: Arguments:
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
