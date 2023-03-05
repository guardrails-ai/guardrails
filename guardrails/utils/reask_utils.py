from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Dict, List, Union

from lxml import etree as ET


@dataclass
class ReAsk:
    incorrect_value: Any
    error_message: str


def gather_reasks(response: Dict) -> List[tuple]:
    """
    Traverse response and gather all ReAsk objects.
    Response is a nested dictionary, where values can also be lists or
    dictionaries.
    Make sure to also grab the corresponding paths (including list index),
    and return a list of tuples.
    """
    reasks = []

    def _gather_reasks(response: Union[list, dict], path: List[str] = []):
        if isinstance(response, dict):
            iterable = response.items()
        elif isinstance(response, list):
            iterable = enumerate(response)
        else:
            raise ValueError(f"Expected dict or list, got {type(response)}")
        for field, value in iterable:
            if isinstance(value, ReAsk):
                reasks.append((path + [field], value))

            if isinstance(value, dict):
                _gather_reasks(value, path + [field])

            if isinstance(value, list):
                for idx, item in enumerate(value):
                    if isinstance(item, ReAsk):
                        reasks.append((path + [field, idx], item))
                    else:
                        _gather_reasks(item, path + [field, idx])

    _gather_reasks(response)
    return reasks



def get_reasks_by_element(
    reasks: List[tuple],
    parsed_aiml: ET._Element,
) -> Dict[ET._Element, List[tuple]]:

    reasks_by_element = defaultdict(list)

    for path, reask in reasks:
        # Make a find query for each path
        # - replace int values in path with '*'
        # TODO: does this work for all cases?
        query = "."
        for part in path:
            if isinstance(part, int):
                query += "/*"
            else:
                query += f"/*[@name='{part}']"

        # Find the element
        element = parsed_aiml.find(query)

        reasks_by_element[element].append((path, reask))

    return reasks_by_element


def get_pruned_tree(
    root: ET._Element,
    reask_elements: List[ET._Element] = None,
) -> str:
    """Prune tree of any elements that are not in `reasks`.

    Return the tree with only the elements that are keys of `reasks` and
    their parents. If `reasks` is None, return the entire tree. If an
    element is removed, remove all ancestors that have no children.

    Args:
        root: The XML tree.
        reasks: The elements that are to be reasked.

    Returns:
        The prompt.
    """
    if reask_elements is None:
        return root

    # Get all elements in `root`
    elements = root.findall(".//*")
    for element in elements:
        if (element not in reask_elements) and len(element) == 0:
            parent = element.getparent()
            parent.remove(element)

            # Remove all ancestors that have no children
            while len(parent) == 0:
                grandparent = parent.getparent()
                grandparent.remove(parent)
                parent = grandparent

    return root


def get_correction_instruction(reasks: List[tuple]) -> str:
    """Construct a correction instruction.

    Args:
        reasks: List of tuples, where each tuple contains the path to the
            reasked element, and the ReAsk object (which contains the error
            message describing why the reask is necessary).
        response: The response.

    Returns:
        The correction instruction.
    """

    correction_instruction = ""

    for reask in reasks:
        reask_path = reask[0]
        reask_element = reask[1]
        reask_error_message = reask_element.error_message
        reask_value = reask_element.incorrect_value

        # Create a helpful prompt that mentions why the reask value is incorrect.
        # The helpful prompt should have english language directions for the user.
        reask_prompt = f" '{reask_value}' is not a valid value for '{reask_path[-1]}' because {reask_error_message}"

        # Add the helpful prompt to the correction instruction.
        correction_instruction += f"{reask_prompt}"

    return correction_instruction