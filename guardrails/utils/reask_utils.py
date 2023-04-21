import json
from collections import defaultdict
from copy import deepcopy
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union

from lxml import etree as ET

from guardrails.prompt import Prompt
from guardrails.utils.constants import constants


@dataclass
class ReAsk:
    incorrect_value: Any
    error_message: str
    fix_value: Any
    path: List[Any] = None


def gather_reasks(validated_output: Dict) -> List[ReAsk]:
    """Traverse output and gather all ReAsk objects.

    Args:
        validated_output (Dict): The output of a model. Each value can be a ReAsk,
            a list, a dictionary, or a single value.

    Returns:
        A list of ReAsk objects found in the output.
    """
    from guardrails.validators import PydanticReAsk

    reasks = []

    def _gather_reasks_in_dict(output: Dict, path: List[str] = []) -> None:
        is_pydantic = isinstance(output, PydanticReAsk)
        for field, value in output.items():
            if isinstance(value, ReAsk):
                if is_pydantic:
                    value.path = path
                else:
                    value.path = path + [field]
                reasks.append(value)

            if isinstance(value, dict):
                _gather_reasks_in_dict(value, path + [field])

            if isinstance(value, list):
                _gather_reasks_in_list(value, path + [field])
        return

    def _gather_reasks_in_list(output: List, path: List[str] = []) -> None:
        for idx, item in enumerate(output):
            if isinstance(item, ReAsk):
                item.path = path + [idx]
                reasks.append(item)
            elif isinstance(item, dict):
                _gather_reasks_in_dict(item, path + [idx])
            elif isinstance(item, list):
                _gather_reasks_in_list(item, path + [idx])
        return

    _gather_reasks_in_dict(validated_output)
    return reasks


def get_reasks_by_element(
    reasks: List[ReAsk],
    parsed_rail: ET._Element,
) -> Dict[ET._Element, List[tuple]]:
    """Cluster reasks by the XML element they are associated with."""
    # This should be guaranteed to work, since the path corresponding
    # to a ReAsk should always be valid in the element tree.

    # This is because ReAsk objects are only created for elements
    # with corresponding validators i.e. the element must have been
    # in the tree in the first place for the ReAsk to be created.

    reasks_by_element = defaultdict(list)

    for reask in reasks:
        path = reask.path
        # TODO: does this work for all cases?
        query = "."
        for part in path:
            if isinstance(part, int):
                query += "/*"
            else:
                query += f"/*[@name='{part}']"

        # Find the element
        element = parsed_rail.find(query)

        reasks_by_element[element].append(reask)

    return reasks_by_element


def get_pruned_tree(
    root: ET._Element,
    reask_elements: List[ET._Element] = None,
) -> ET._Element:
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

    pruned_elements = root.findall(".//*")
    for element in pruned_elements:
        if element not in reask_elements:
            # Remove the format attribute
            if "format" in element.attrib:
                del element.attrib["format"]

    return root


def prune_json_for_reasking(json_object: Any) -> Union[None, Dict, List]:
    """Validated JSON is a nested dictionary where some keys may be ReAsk
    objects.

    This function prunes the validated JSON of any object that is not a ReAsk object.
    It also keeps all of the ancestors of the ReAsk objects.

    Args:
        json_object: The validated JSON.

    Returns:
        The pruned validated JSON.
    """
    from guardrails.validators import PydanticReAsk

    if isinstance(json_object, ReAsk) or isinstance(json_object, PydanticReAsk):
        return json_object
    elif isinstance(json_object, list):
        pruned_list = []
        for item in json_object:
            pruned_output = prune_json_for_reasking(item)
            if pruned_output is not None:
                pruned_list.append(pruned_output)
        if len(pruned_list):
            return pruned_list
        return None
    elif isinstance(json_object, dict):
        pruned_json = {}
        for key, value in json_object.items():
            if isinstance(value, ReAsk) or isinstance(value, PydanticReAsk):
                pruned_json[key] = value
            elif isinstance(value, dict):
                pruned_output = prune_json_for_reasking(value)
                if pruned_output is not None:
                    pruned_json[key] = pruned_output
            elif isinstance(value, list):
                pruned_list = []
                for item in value:
                    pruned_output = prune_json_for_reasking(item)
                    if pruned_output is not None:
                        pruned_list.append(pruned_output)
                if len(pruned_list):
                    pruned_json[key] = pruned_list

        if len(pruned_json):
            return pruned_json

        return None


def get_reask_prompt(
    parsed_rail: ET._Element,
    reasks: List[ReAsk],
    reask_json: Dict,
    reask_prompt_template: Optional[Prompt] = None,
) -> Tuple[Prompt, ET._Element]:
    """Construct a prompt for reasking.

    Args:
        parsed_rail: The parsed RAIL.
        reasks: List of tuples, where each tuple contains the path to the
            reasked element, and the ReAsk object (which contains the error
            message describing why the reask is necessary).
        reask_json: Pruned JSON that contains only ReAsk objects.

    Returns:
        The prompt.
    """
    from guardrails.schema import OutputSchema

    parsed_rail_copy = deepcopy(parsed_rail)

    # Get the elements that are to be reasked
    reask_elements = get_reasks_by_element(reasks, parsed_rail_copy)

    # Get the pruned JSON so that it only contains ReAsk objects
    # Get the pruned tree
    pruned_tree = get_pruned_tree(parsed_rail_copy, list(reask_elements.keys()))
    pruned_tree_string = OutputSchema(pruned_tree).transpile()

    if reask_prompt_template is None:
        reask_prompt_template = Prompt(
            constants["high_level_reask_prompt"] + constants["complete_json_suffix"]
        )

    def reask_decoder(obj):
        return {k: v for k, v in obj.__dict__.items() if k not in ["path", "fix_value"]}

    reask_prompt = reask_prompt_template.format(
        previous_response=json.dumps(reask_json, indent=2, default=reask_decoder)
        .replace("{", "{{")
        .replace("}", "}}"),
        output_schema=pruned_tree_string,
    )

    return reask_prompt, pruned_tree


def reask_json_as_dict(json: Dict) -> Dict:
    """If a ReAsk object exists in the JSON, return it as a dictionary."""

    def _reask_json_as_dict(json_object: Any) -> Any:
        if isinstance(json_object, dict):
            return {
                key: _reask_json_as_dict(value) for key, value in json_object.items()
            }
        elif isinstance(json_object, list):
            return [_reask_json_as_dict(item) for item in json_object]
        elif isinstance(json_object, ReAsk):
            return json_object.__dict__
        else:
            return json_object

    return _reask_json_as_dict(json)


def sub_reasks_with_fixed_values(value: Any) -> Any:
    """Substitute ReAsk objects with their fixed values recursively.

    Args:
        value: Either a list, a dictionary, a ReAsk object or a scalar value.

    Returns:
        The value with ReAsk objects replaced with their fixed values.
    """
    if isinstance(value, list):
        for index, item in enumerate(value):
            value[index] = sub_reasks_with_fixed_values(item)
    elif isinstance(value, dict):
        for dict_key, dict_value in value.items():
            value[dict_key] = sub_reasks_with_fixed_values(dict_value)
    elif isinstance(value, ReAsk):
        value = value.fix_value

    return value
