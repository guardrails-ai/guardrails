from collections import defaultdict
from typing import Any, Dict, List, Union

import pydantic
from lxml import etree as ET

from guardrails.validators import FailResult


class ReAsk(pydantic.BaseModel):
    incorrect_value: Any
    fail_results: List[FailResult]


class FieldReAsk(ReAsk):
    path: List[Any] = None


class SkeletonReAsk(ReAsk):
    pass


class NonParseableReAsk(ReAsk):
    pass


def gather_reasks(validated_output: Dict) -> List[FieldReAsk]:
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
            if isinstance(value, FieldReAsk):
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
            if isinstance(item, FieldReAsk):
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
    reasks: List[FieldReAsk],
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
            while parent is not None and len(parent) == 0:
                grandparent = parent.getparent()
                if grandparent is not None:
                    grandparent.remove(parent)
                parent = grandparent

    pruned_elements = root.findall(".//*")
    for element in pruned_elements:
        if element not in reask_elements:
            # Remove the format attribute
            if "format" in element.attrib:
                del element.attrib["format"]

    return root


def prune_obj_for_reasking(obj: Any) -> Union[None, Dict, List]:
    """After validation, we get a nested dictionary where some keys may be
    ReAsk objects.

    This function prunes the validated form of any object that is not a ReAsk object.
    It also keeps all of the ancestors of the ReAsk objects.

    Args:
        obj: The validated object.

    Returns:
        The pruned validated object.
    """
    from guardrails.validators import PydanticReAsk

    if isinstance(obj, ReAsk) or isinstance(obj, PydanticReAsk):
        return obj
    elif isinstance(obj, list):
        pruned_list = []
        for item in obj:
            pruned_output = prune_obj_for_reasking(item)
            if pruned_output is not None:
                pruned_list.append(pruned_output)
        if len(pruned_list):
            return pruned_list
        return None
    elif isinstance(obj, dict):
        pruned_json = {}
        for key, value in obj.items():
            if isinstance(value, FieldReAsk) or isinstance(value, PydanticReAsk):
                pruned_json[key] = value
            elif isinstance(value, dict):
                pruned_output = prune_obj_for_reasking(value)
                if pruned_output is not None:
                    pruned_json[key] = pruned_output
            elif isinstance(value, list):
                pruned_list = []
                for item in value:
                    pruned_output = prune_obj_for_reasking(item)
                    if pruned_output is not None:
                        pruned_list.append(pruned_output)
                if len(pruned_list):
                    pruned_json[key] = pruned_list

        if len(pruned_json):
            return pruned_json

        return None


def reasks_to_dict(dict_with_reasks: Dict) -> Dict:
    """If a ReAsk object exists in the dict, return it as a dictionary."""

    def _(dict_object: Any) -> Any:
        if isinstance(dict_object, dict):
            return {key: _(value) for key, value in dict_object.items()}
        elif isinstance(dict_object, list):
            return [_(item) for item in dict_object]
        elif isinstance(dict_object, FieldReAsk):
            return dict_object.__dict__
        else:
            return dict_object

    return _(dict_with_reasks)


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
    elif isinstance(value, FieldReAsk):
        # TODO handle multiple fail results
        value = value.fail_results[0].fix_value

    return value
