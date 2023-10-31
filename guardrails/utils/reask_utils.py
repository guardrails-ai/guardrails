from typing import Any, Dict, List, Optional, Union

import pydantic

from guardrails.datatypes import List as ListType
from guardrails.datatypes import Object as ObjectType
from guardrails.validator_base import FailResult


class ReAsk(pydantic.BaseModel):
    incorrect_value: Any
    fail_results: List[FailResult]


class FieldReAsk(ReAsk):
    path: Optional[List[Any]] = None


class SkeletonReAsk(ReAsk):
    pass


class NonParseableReAsk(ReAsk):
    pass


def gather_reasks(validated_output: Optional[Union[Dict, ReAsk]]) -> List[ReAsk]:
    """Traverse output and gather all ReAsk objects.

    Args:
        validated_output (Dict): The output of a model. Each value can be a ReAsk,
            a list, a dictionary, or a single value.

    Returns:
        A list of ReAsk objects found in the output.
    """
    if validated_output is None:
        return []
    if isinstance(validated_output, ReAsk):
        return [validated_output]

    reasks = []

    def _gather_reasks_in_dict(
        output: Dict, path: Optional[List[Union[str, int]]] = None
    ) -> None:
        if path is None:
            path = []
        for field, value in output.items():
            if isinstance(value, FieldReAsk):
                value.path = path + [field]
                reasks.append(value)

            if isinstance(value, dict):
                _gather_reasks_in_dict(value, path + [field])

            if isinstance(value, list):
                _gather_reasks_in_list(value, path + [field])
        return

    def _gather_reasks_in_list(
        output: List, path: Optional[List[Union[str, int]]] = None
    ) -> None:
        if path is None:
            path = []
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


def get_pruned_tree(
    root: ObjectType,
    reasks: Optional[List[FieldReAsk]] = None,
) -> ObjectType:
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
    if reasks is None:
        return root

    # Find all elements that are to be retained
    retain = [root]
    for reask in reasks:
        path = reask.path
        if path is None:
            raise RuntimeError("FieldReAsk path is None")
        current_root = root
        for part in path:
            # TODO does this work for all cases?
            if isinstance(part, int):
                current_root = current_root.children.item
            else:
                current_root = vars(current_root.children)[part]
            retain.append(current_root)

    # Remove all elements that are not to be retained
    def _remove_children(element: Union[ObjectType, ListType]) -> None:
        if isinstance(element, ListType):
            if element.children.item not in retain:
                del element._children["item"]
            else:
                _remove_children(element.children.item)
        else:  # if isinstance(element, ObjectType):
            for child_name, child in vars(element.children).items():
                if child not in retain:
                    del element._children[child_name]
                else:
                    _remove_children(child)

    _remove_children(root)

    return root


def prune_obj_for_reasking(obj: Any) -> Union[None, Dict, List, ReAsk]:
    """After validation, we get a nested dictionary where some keys may be
    ReAsk objects.

    This function prunes the validated form of any object that is not a ReAsk object.
    It also keeps all of the ancestors of the ReAsk objects.

    Args:
        obj: The validated object.

    Returns:
        The pruned validated object.
    """

    if isinstance(obj, ReAsk):
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
            if isinstance(value, FieldReAsk):
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
