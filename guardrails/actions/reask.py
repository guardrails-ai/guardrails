from copy import deepcopy
import json
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

from guardrails_api_client import Reask as IReask
from guardrails.classes.execution.guard_execution_options import GuardExecutionOptions
from guardrails.classes.output_type import OutputTypes
from guardrails.classes.validation.validation_result import FailResult
from guardrails.prompt.instructions import Instructions
from guardrails.prompt.prompt import Prompt
from guardrails.schema.generator import generate_example
from guardrails.schema.rail_schema import json_schema_to_rail_output
from guardrails.types.validator import ValidatorMap
from guardrails.utils.constants import constants
from guardrails.utils.prompt_utils import prompt_content_for_schema, prompt_uses_xml


### Classes/Types ###
class ReAsk(IReask):
    """Base class for ReAsk objects.

    Attributes:
        incorrect_value (Any): The value that failed validation.
        fail_results (List[FailResult]): The results of the failed validations.
    """

    incorrect_value: Any
    fail_results: List[FailResult]

    @classmethod
    def from_interface(cls, reask: IReask) -> "ReAsk":
        fail_results = []
        if reask.fail_results:
            fail_results: List[FailResult] = [
                FailResult.from_interface(fail_result)
                for fail_result in reask.fail_results
            ]

        if reask.additional_properties.get("path"):
            return FieldReAsk(
                incorrect_value=reask.incorrect_value,
                fail_results=fail_results,
                path=reask.additional_properties["path"],
            )

        if len(fail_results) == 1:
            error_message = fail_results[0].error_message
            if error_message == "Output is not parseable as JSON":
                return NonParseableReAsk(
                    incorrect_value=reask.incorrect_value,
                    fail_results=fail_results,
                )
            elif "JSON does not match schema" in error_message:
                return SkeletonReAsk(
                    incorrect_value=reask.incorrect_value,
                    fail_results=fail_results,
                )

        return cls(incorrect_value=reask.incorrect_value, fail_results=fail_results)

    @classmethod
    def from_dict(cls, obj: Dict[str, Any]) -> Optional["ReAsk"]:
        i_reask = super().from_dict(obj)
        if not i_reask:
            return None
        return cls.from_interface(i_reask)


class FieldReAsk(ReAsk):
    """An implementation of ReAsk that is used to reask for a specific field.
    Inherits from ReAsk.

    Attributes:
        path (Optional[List[Any]]): a list of keys that
            designated the path to the field that failed validation.
    """

    # FIXME: This shouldn't be optional
    # We should be able to assign it on init now
    path: Optional[List[Any]] = None


class SkeletonReAsk(ReAsk):
    """An implementation of ReAsk that is used to reask for structured data
    when the response does not match the expected schema.

    Inherits from ReAsk.
    """

    pass


class NonParseableReAsk(ReAsk):
    """An implementation of ReAsk that is used to reask for structured data
    when the response is not parseable as JSON.

    Inherits from ReAsk.
    """

    pass


### Internal Helper Methods ###
def get_reask_subschema(
    json_schema: Dict[str, Any],
    reasks: Optional[List[FieldReAsk]] = None,
) -> Dict[str, Any]:
    """Prune schema of any subschemas that are not in `reasks`.

    Return the schema with only the subschemas that are being `reask`ed for and
    their parents. If `reasks` is None, return the entire schema. If an
    subschema is removed, remove all ancestors that have no children.

    Args:
        root: A JSON Schema
        reasks: The fields that are to be reasked.

    Returns:
        A JSON Schema.
    """
    root = deepcopy(json_schema)

    if reasks is None:
        return root

    # Find all elements that are to be retained
    # NOTE: At this point, in the case of discriminated unions,
    #   the LLM has already decided which subschema of the union to use.
    #   This means that we can flatten complex schema compositions, e.g. anyOf's,
    #   and just build a subschema that represents the resolved schema
    #   of the LLM response.
    # schema_paths_to_retain = []
    # for reask in reasks:
    #     path = reask.path
    #     if path is None:
    #         raise RuntimeError("FieldReAsk path is None")
    #     schema_path = "$"
    #     for part in path:
    #         if isinstance(part, int):
    #             schema_path += ".items"
    #         else:
    #             schema_path += f".properties.{path}"
    #     schema_paths_to_retain.append(schema_path)

    # # Remove all elements that are not to be retained
    # def _prune_schema(schema: Dict[str, Any]) -> None:
    #     if schema.get("type") == SimpleTypes.ARRAY:
    #         if schema.children.item not in retain:
    #             del schema._children["item"]
    #         else:
    #             _prune_schema(schema.children.item)
    #     else:  # if isinstance(schema, ObjectType):
    #         for child_name, child in vars(schema.children).items():
    #             if child not in retain:
    #                 del schema._children[child_name]
    #             else:
    #                 _prune_schema(child)

    # _prune_schema(root)

    # FIXME: PUNT
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


def update_response_by_path(output: dict, path: List[Any], value: Any) -> None:
    """Update the output by path.

    Args:
        output: The output.
        path: The path to the element to be updated.
        value: The value to be updated.
    """
    for key in path[:-1]:
        output = output[key]
    output[path[-1]] = value


### Guard Execution Methods ###
def introspect(
    data: Optional[Union[ReAsk, str, Dict, List]],
) -> Tuple[Sequence[ReAsk], Optional[Union[str, Dict, List]]]:
    if isinstance(data, FieldReAsk):
        return [data], None
    elif isinstance(data, SkeletonReAsk):
        return [data], None
    elif isinstance(data, NonParseableReAsk):
        return [data], None
    return gather_reasks(data)


def get_reask_setup_for_string(
    output_type: OutputTypes,
    output_schema: Dict[str, Any],
    validation_map: ValidatorMap,
    reasks: Sequence[ReAsk],
    *,
    validation_response: Optional[Union[str, List, Dict, ReAsk]] = None,
    prompt_params: Optional[Dict[str, Any]] = None,
    exec_options: Optional[GuardExecutionOptions] = None,
) -> Tuple[Dict[str, Any], Prompt, Instructions]:
    prompt_params = prompt_params or {}
    exec_options = exec_options or GuardExecutionOptions()

    schema_prompt_content = prompt_content_for_schema(
        output_type, output_schema, validation_map
    )
    xml_output_schema = json_schema_to_rail_output(
        json_schema=output_schema, validator_map=validation_map
    )

    reask_prompt_template = None
    if exec_options.reask_prompt:
        reask_prompt_template = Prompt(exec_options.reask_prompt)
    else:
        reask_prompt_template = Prompt(
            constants["high_level_string_reask_prompt"]
            + constants["complete_string_suffix"]
        )

    error_messages = "\n".join(
        [
            f"- {fail_result.error_message}"
            for reask in reasks
            for fail_result in reask.fail_results
        ]
    )

    prompt = reask_prompt_template.format(
        # FIXME: How do we properly type this?
        # Solution will have to come from Runner all the way down to here
        previous_response=validation_response.incorrect_value,  # type: ignore
        error_messages=error_messages,
        output_schema=schema_prompt_content,
        xml_output_schema=xml_output_schema,
        **prompt_params,
    )

    instructions = None
    if exec_options.reask_instructions:
        instructions = Instructions(exec_options.reask_instructions)
    if instructions is None:
        instructions = Instructions("You are a helpful assistant.")
    instructions = instructions.format(
        output_schema=schema_prompt_content,
        xml_output_schema=xml_output_schema,
        **prompt_params,
    )

    return output_schema, prompt, instructions


def get_original_prompt(exec_options: Optional[GuardExecutionOptions] = None) -> str:
    exec_options = exec_options or GuardExecutionOptions()
    original_msg_history = exec_options.msg_history or []
    msg_history_prompt = next(
        (
            h.get("content")
            for h in original_msg_history
            if isinstance(h, dict) and h.get("role") == "user"
        ),
        "",
    )
    original_prompt = exec_options.prompt or msg_history_prompt or ""
    return original_prompt


def get_reask_setup_for_json(
    output_type: OutputTypes,
    output_schema: Dict[str, Any],
    validation_map: ValidatorMap,
    reasks: Sequence[ReAsk],
    *,
    parsing_response: Optional[Union[str, List, Dict, ReAsk]] = None,
    validation_response: Optional[Union[str, List, Dict, ReAsk]] = None,
    use_full_schema: Optional[bool] = False,
    prompt_params: Optional[Dict[str, Any]] = None,
    exec_options: Optional[GuardExecutionOptions] = None,
) -> Tuple[Dict[str, Any], Prompt, Instructions]:
    reask_schema = output_schema
    is_skeleton_reask = not any(isinstance(reask, FieldReAsk) for reask in reasks)
    is_nonparseable_reask = any(
        isinstance(reask, NonParseableReAsk) for reask in reasks
    )
    error_messages = {}
    prompt_params = prompt_params or {}
    exec_options = exec_options or GuardExecutionOptions()
    original_prompt = get_original_prompt(exec_options)
    use_xml = prompt_uses_xml(original_prompt)

    reask_prompt_template = None
    if exec_options.reask_prompt:
        reask_prompt_template = Prompt(exec_options.reask_prompt)

    if is_nonparseable_reask:
        if reask_prompt_template is None:
            suffix = (
                constants["xml_suffix_without_examples"]
                if use_xml
                else constants["json_suffix_without_examples"]
            )
            reask_prompt_template = Prompt(
                constants["high_level_json_parsing_reask_prompt"] + suffix
            )
        np_reask: NonParseableReAsk = next(
            r for r in reasks if isinstance(r, NonParseableReAsk)
        )
        # Give the LLM what it gave us that couldn't be parsed as JSON
        reask_value = np_reask.incorrect_value
    elif is_skeleton_reask:
        if reask_prompt_template is None:
            reask_prompt = constants["high_level_skeleton_reask_prompt"]

            if use_xml:
                reask_prompt = (
                    reask_prompt + constants["xml_suffix_with_structure_example"]
                )
            else:
                reask_prompt = (
                    reask_prompt
                    + constants["error_messages"]
                    + constants["json_suffix_with_structure_example"]
                )

            reask_prompt_template = Prompt(reask_prompt)

        # Validation hasn't happend yet
        #   and the problem is with the json the LLM gave us.
        # Give it this same json and tell it to fix it.
        reask_value = validation_response if use_xml else parsing_response
        skeleton_reask: SkeletonReAsk = next(
            r for r in reasks if isinstance(r, SkeletonReAsk)
        )
        error_messages = skeleton_reask.fail_results[0].error_message
    else:
        if use_full_schema:
            # Give the LLM the full JSON that failed validation
            reask_value = validation_response if use_xml else parsing_response
            # Don't prune the tree if we're reasking with pydantic model
            # (and openai function calling)
        else:
            # Prune out the individual fields that did not fail validation.
            # Only reask for field that did fail.
            reask_value = prune_obj_for_reasking(validation_response)

            # Generate a subschema that matches the specific fields we're reasking for.
            field_reasks = [r for r in reasks if isinstance(r, FieldReAsk)]
            reask_schema = get_reask_subschema(output_schema, field_reasks)

        if reask_prompt_template is None:
            suffix = (
                constants["xml_suffix_without_examples"]
                if use_xml
                else constants["json_suffix_without_examples"]
            )
            reask_prompt_template = Prompt(
                constants["high_level_json_reask_prompt"] + suffix
            )

        error_messages = {
            ".".join(str(p) for p in r.path): "; ".join(  # type: ignore
                f.error_message for f in r.fail_results
            )
            for r in reasks
            if isinstance(r, FieldReAsk)
        }

    stringified_schema = prompt_content_for_schema(
        output_type, reask_schema, validation_map
    )
    xml_output_schema = json_schema_to_rail_output(
        json_schema=output_schema, validator_map=validation_map
    )

    json_example = json.dumps(
        generate_example(reask_schema),
        indent=2,
    )

    def reask_decoder(obj: ReAsk):
        decoded = {}
        for k, v in obj.__dict__.items():
            if k in ["path", "additional_properties"]:
                continue
            if k == "fail_results":
                k = "error_messages"
                v = [result.error_message for result in v]
            decoded[k] = v
        return decoded

    prompt = reask_prompt_template.format(
        previous_response=json.dumps(
            reask_value, indent=2, default=reask_decoder, ensure_ascii=False
        ),
        output_schema=stringified_schema,
        xml_output_schema=xml_output_schema,
        json_example=json_example,
        error_messages=json.dumps(error_messages),
        **prompt_params,
    )

    instructions = None
    if exec_options.reask_instructions:
        instructions = Instructions(exec_options.reask_instructions)
    else:
        instructions_const = (
            constants["high_level_xml_instructions"]
            if use_xml
            else constants["high_level_json_instructions"]
        )
        instructions = Instructions(instructions_const)
    instructions = instructions.format(**prompt_params)

    return reask_schema, prompt, instructions


def get_reask_setup(
    output_type: OutputTypes,
    output_schema: Dict[str, Any],
    validation_map: ValidatorMap,
    reasks: Sequence[ReAsk],
    *,
    parsing_response: Optional[Union[str, List, Dict, ReAsk]] = None,
    validation_response: Optional[Union[str, List, Dict, ReAsk]] = None,
    use_full_schema: Optional[bool] = False,
    prompt_params: Optional[Dict[str, Any]] = None,
    exec_options: Optional[GuardExecutionOptions] = None,
) -> Tuple[Dict[str, Any], Prompt, Instructions]:
    prompt_params = prompt_params or {}
    exec_options = exec_options or GuardExecutionOptions()

    if output_type == OutputTypes.STRING:
        return get_reask_setup_for_string(
            output_type=output_type,
            output_schema=output_schema,
            validation_map=validation_map,
            reasks=reasks,
            validation_response=validation_response,
            prompt_params=prompt_params,
            exec_options=exec_options,
        )
    return get_reask_setup_for_json(
        output_type=output_type,
        output_schema=output_schema,
        validation_map=validation_map,
        reasks=reasks,
        parsing_response=parsing_response,
        validation_response=validation_response,
        use_full_schema=use_full_schema,
        prompt_params=prompt_params,
        exec_options=exec_options,
    )


### Post-Processing Methods ###
def gather_reasks(
    validated_output: Optional[Union[ReAsk, str, Dict, List]],
) -> Tuple[List[ReAsk], Optional[Union[str, Dict, List]]]:
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
    if isinstance(validated_output, str):
        return [], validated_output

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
    elif isinstance(validated_output, List):
        valid_output = deepcopy(validated_output)
        _gather_reasks_in_list(validated_output, valid_output)
        return reasks, valid_output
    return reasks, None


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


def merge_reask_output(previous_response, reask_response) -> Dict:
    """Merge the reask output into the original output.

    Args:
        prev_logs: validation output object from the previous iteration.
        current_logs: validation output object from the current iteration.

    Returns:
        The merged output.
    """
    if isinstance(previous_response, ReAsk):
        return reask_response

    # FIXME: Uncommenet when field level reask is fixed
    # This used to be necessary for field level reask because
    #   the schema was pruned to only the properties that failed.
    #  This caused previous keys that were correct to be pruned during schemafication.
    # pruned_reask_json = prune_obj_for_reasking(previous_response)
    pruned_reask_json = previous_response

    # Reask output and reask json have the same structure, except that values
    # of the reask json are ReAsk objects. We want to replace the ReAsk objects
    # with the values from the reask output.
    merged_json = deepcopy(previous_response)

    def update_reasked_elements(pruned_reask_json, reask_response_dict):
        if isinstance(pruned_reask_json, dict):
            for key, value in pruned_reask_json.items():
                if isinstance(value, FieldReAsk):
                    if value.path is None:
                        raise RuntimeError(
                            "FieldReAsk object must have a path attribute."
                        )
                    corrected_value = reask_response_dict.get(key)
                    update_response_by_path(merged_json, value.path, corrected_value)
                else:
                    update_reasked_elements(
                        pruned_reask_json[key], reask_response_dict[key]
                    )
        elif isinstance(pruned_reask_json, list):
            for i, item in enumerate(pruned_reask_json):
                if isinstance(item, FieldReAsk):
                    if item.path is None:
                        raise RuntimeError(
                            "FieldReAsk object must have a path attribute."
                        )
                    corrected_value = reask_response_dict[i]
                    update_response_by_path(merged_json, item.path, corrected_value)
                else:
                    update_reasked_elements(
                        pruned_reask_json[i], reask_response_dict[i]
                    )

    update_reasked_elements(pruned_reask_json, reask_response)

    return merged_json
