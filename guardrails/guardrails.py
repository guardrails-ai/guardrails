import json
import logging
from copy import deepcopy
from typing import Any, Dict, List, Tuple, Callable

from eliot import start_action, to_file

import guardrails.utils.reask_utils as reask_utils
from guardrails.llm_providers import get_llm_ask
from guardrails.prompt import Prompt
from guardrails.output_schema import OutputSchema
from guardrails.utils.rail_utils import read_rail
from guardrails.utils.logs_utils import GuardHistory, GuardLogs, GuardState

logger = logging.getLogger(__name__)
to_file(open("guardrails.log", "w"))


class Guard:
    def __init__(
        self,
        schema: OutputSchema,
        base_prompt: Prompt,
        num_reasks: int = 1,
    ):
        self.output_schema = schema
        self.num_reasks = num_reasks
        self.guard_state = GuardState([])

        parsed_rail_copy = deepcopy(self.output_schema.parsed_rail)
        output_schema_prompt = reask_utils.extract_prompt_from_xml(parsed_rail_copy)
        self.base_prompt = base_prompt.format(output_schema=output_schema_prompt)

    @classmethod
    def from_rail(cls, rail_file: str) -> "Guard":
        """Create an Schema from an XML file."""
        output_schema, base_prompt, _ = read_rail(rail_file=rail_file)
        return cls(output_schema, base_prompt)

    @classmethod
    def from_rail_string(cls, rail_string: str) -> "Guard":
        """Create an Schema from an XML string."""
        output_schema, base_prompt, _ = read_rail(rail_string=rail_string)
        return cls(output_schema, base_prompt)

    def __call__(
        self, llm_api: Callable, prompt_params: Dict = None, *args, **kwargs
    ) -> Tuple[str, Dict]:
        """Outermost function that calls the LLM and validates the output.

        Args:
            llm_api: The LLM API to call (e.g. openai.Completion.create)
            prompt_params: The parameters to pass to the prompt.format() method.
            *args: Additional arguments to pass to the LLM API.
            **kwargs: Additional keyword arguments to pass to the LLM API.

        Returns:
            The raw text output from the LLM and the validated output.
        """
        with start_action(action_type="guard_call", prompt_params=prompt_params):
            prompt = self.base_prompt
            if prompt_params is None:
                prompt_params = {}
            prompt = self.base_prompt.format(**prompt_params)
            llm_ask = get_llm_ask(llm_api, *args, **kwargs)

            return self.ask_with_validation(prompt, llm_ask)

    def ask_with_validation(
        self, prompt: str, llm_ask: Callable
    ) -> Tuple[str, Dict]:
        """Ask a question, and validate the output."""

        with start_action(action_type="ask_with_validation", prompt=prompt):
            guard_history = self.validation_inner_loop(
                prompt, llm_ask, 0, self.output_schema
            )

            return (
                guard_history.output,
                guard_history.validated_response,
            )

    def validation_inner_loop(
        self,
        formatted_prompt: str,
        llm_ask: Callable,
        reask_ctr: int,
        output_schema: OutputSchema,
        guard_history: GuardHistory = None,
    ) -> GuardHistory:
        """
        Ask a question, and validate the output.

        Args:
            formatted_prompt: The prompt to send to the LLM.
            reask_ctr: The number of times the user has been reasked.

        Returns:
            The raw output from the LLM, the output as a dict, and the
            validated output.
        """

        if guard_history is None:
            guard_history = GuardHistory([])

        with start_action(
            action_type="validation_inner_loop", reask_ctr=reask_ctr
        ) as action:

            output = llm_ask(formatted_prompt)
            action.log(message_type="info", prompt=formatted_prompt, output=output)

            try:
                output_as_dict = json.loads(output)
                action.log(message_type="info", output_as_dict=output_as_dict)
                validated_response, reasks = self.validate_output(
                    output_as_dict, output_schema
                )
            except json.decoder.JSONDecodeError:
                validated_response = None
                output_as_dict = None
                reasks = []
                action.log(message_type="info", output_as_dict=output_as_dict)

            action.log(
                message_type="info",
                validated_response=reask_utils.reask_json_as_dict(validated_response),
                reasks=[r.__dict__ for r in reasks],
            )

            gd_log = GuardLogs(
                prompt=formatted_prompt,
                output=output,
                output_as_dict=output_as_dict,
                validated_response=validated_response,
                reasks=reasks,
            )

            guard_history = guard_history.push(gd_log)

            if len(reasks) and reask_ctr < self.num_reasks:

                reask_json = reask_utils.prune_json_for_reasking(validated_response)
                reask_prompt, reask_schema = reask_utils.get_reask_prompt(
                    self.output_schema.parsed_rail, reasks, reask_json
                )

                return self.validation_inner_loop(
                    reask_prompt, llm_ask, reask_ctr + 1, reask_schema, guard_history
                )

            self.guard_state = self.guard_state.push(guard_history)

            return guard_history

    def validate_output(
        self, output: Dict[str, Any], schema: OutputSchema
    ) -> Tuple[Dict[str, Any], List[reask_utils.ReAsk]]:
        """Validate a output against the schema.

        Args:
            output: The output to validate.

        Returns:
            Tuple, where the first element is the validated output, and the
            second element is a list of tuples, where each tuple contains the
            path to the reasked element, and the ReAsk object.
        """

        validated_response = deepcopy(output)

        for field, value in validated_response.items():
            if field not in schema:
                logger.debug(f"Field {field} not in schema.")
                continue

            validated_response = schema[field].validate(
                field, value, validated_response
            )

        reasks = reask_utils.gather_reasks(validated_response)

        return (validated_response, reasks)

    def __repr__(self):
        def _print_dict(d: Dict[str, Any], indent: int = 0) -> str:
            """Print a dictionary in a nice way."""

            s = ""
            for k, v in d.items():
                if isinstance(v, dict):
                    s += f"{k}:\n{_print_dict(v, indent=indent + 1)}"
                else:
                    s += f"{' ' * (indent * 4)}{k}: {v}\n"

            return s

        schema = _print_dict(self.output_schema)

        return f"Schema({schema})"
