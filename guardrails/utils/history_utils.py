from typing import Dict, Union

from guardrails.classes.history import Call
from guardrails.utils.logs_utils import merge_reask_output
from guardrails.utils.reask_utils import ReAsk


def merge_validation_output(
    current_call: Call,
    is_full_schema_reask: bool
) -> Union[ReAsk, str, Dict]:
        number_of_iterations = current_call.iterations.length
        
        # Don't try to merge if
        #   1. We plan to perform full schema reasks
        #   2. There's nothing to merge
        #   3. The output is a top level ReAsk (i.e. SkeletonReAsk or NonParseableReask)
        #   4. The output is a string
        if (
            is_full_schema_reask
            or number_of_iterations < 2
            or isinstance(current_call.iterations.last.validation_output, ReAsk)
            or isinstance(current_call.iterations.last.validation_output, str)
        ):
            return current_call.iterations.last.validation_output
        
        
        current_index = 1
        merged_validation_output = current_call.iterations.first.validation_output
        while current_index < number_of_iterations:
            current_validation_output = current_call.iterations.at(current_index).validation_output
            merged_validation_output = merge_reask_output(
                merged_validation_output,
                current_validation_output
            )
            current_index = current_index + 1

        return merged_validation_output