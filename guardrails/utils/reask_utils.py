from copy import deepcopy
from typing import Any, Dict, List, Optional, Tuple, Union

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


def gather_reasks(
    validated_output: Optional[Union[str, Dict, ReAsk]]
) -> Tuple[List[ReAsk], Optional[Dict]]:
    """Traverse output and gather all ReAsk objects.

    Args:
        validated_output (Union[str, Dict, ReAsk], optional): The output of a model.
            Each value can be a ReAsk, a list, a dictionary, or a single value.

    Returns:
        A list of ReAsk objects found in the output.
    """
    if validated_output is None:
        return [], None
    if isinstance(validated_output, ReAsk):
        return [validated_output], None

    reasks = []

    def _gather_reasks_in_dict(
        original: Dict, valid_output: Dict, path: Optional[List[Union[str, int]]] = None
    ) -> None:
        if path is None:
            path = []
        for field, value in original.items():
            if isinstance(value, FieldReAsk):
                value.path = path + [field]
                reasks.append(value)
                del valid_output[field]

            if isinstance(value, dict):
                _gather_reasks_in_dict(value, valid_output[field], path + [field])

            if isinstance(value, list):
                _gather_reasks_in_list(value, valid_output[field], path + [field])
        return

    def _gather_reasks_in_list(
        original: List, valid_output: List, path: Optional[List[Union[str, int]]] = None
    ) -> None:
        if path is None:
            path = []
        for idx, item in enumerate(original):
            if isinstance(item, FieldReAsk):
                item.path = path + [idx]
                reasks.append(item)
                del valid_output[idx]
            elif isinstance(item, dict):
                _gather_reasks_in_dict(item, valid_output[idx], path + [idx])
            elif isinstance(item, list):
                _gather_reasks_in_list(item, valid_output[idx], path + [idx])
        return

    if isinstance(validated_output, Dict):
        valid_output = deepcopy(validated_output)
        _gather_reasks_in_dict(validated_output, valid_output)
        return reasks, valid_output
    return reasks, None


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
    copy = deepcopy(value)
    if isinstance(copy, list):
        for index, item in enumerate(copy):
            copy[index] = sub_reasks_with_fixed_values(item)
    elif isinstance(copy, dict):
        for dict_key, dict_value in value.items():
            copy[dict_key] = sub_reasks_with_fixed_values(dict_value)
    elif isinstance(copy, FieldReAsk):
        fix_value = copy.fail_results[0].fix_value
        # TODO handle multiple fail results
        # Leave the ReAsk in place if there is no fix value
        # This allows us to determine the proper status for the call
        copy = fix_value if fix_value is not None else copy

    return copy
