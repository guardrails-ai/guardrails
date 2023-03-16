import json
import logging
from copy import deepcopy
from typing import Any, Callable, Dict, List, Optional, Tuple

from eliot import start_action, to_file

import guardrails.utils.reask_utils as reask_utils
from guardrails.llm_providers import get_llm_ask
from guardrails.output_schema import OutputSchema
from guardrails.prompt import Prompt
from guardrails.utils.logs_utils import GuardHistory, GuardLogs, GuardState
from guardrails.utils.rail_utils import read_rail
from guardrails.validators import check_refrain_in_dict, filter_in_dict

logger = logging.getLogger(__name__)
to_file(open("guardrails.log", "w"))


class Guard:
    """The Guard class.

    This class is the main entry point for using Guardrails. It is
    initialized from either `from_rail` or `from_rail_string` methods,
    which take in a `.rail` file or string, respectively. The `__call__`
    method functions as a wrapper around LLM APIs. It takes in an LLM
    API, and optional prompt parameters, and returns the raw output from
    the LLM and the validated output.
    """

    def __init__(
        self,
        schema: OutputSchema,
        base_prompt: Prompt,
        num_reasks: int = 1,
    ):
        self.output_schema = schema
        self.num_reasks = num_reasks
        self.guard_state = GuardState([])

        # raw_prompt is an instance of the Prompt class.
        self.raw_prompt = base_prompt
        # base_prompt is a string, and contains output schema instructions.
        self.base_prompt = base_prompt.source

    @classmethod
    def from_rail(cls, rail_file: str, num_reasks: int = 1) -> "Guard":
        """Create an Schema from an `.rail` file.

        Args:
            rail_file: The path to the `.rail` file.
            num_reasks: The max times to re-ask the LLM for invalid output.

        Returns:
            An instance of the `Guard` class.
        """
        output_schema, base_prompt, _ = read_rail(rail_file=rail_file)
        return cls(output_schema, base_prompt, num_reasks=num_reasks)

    @classmethod
    def from_rail_string(cls, rail_string: str, num_reasks: int = 1) -> "Guard":
        """Create an Schema from an `.rail` string.

        Args:
            rail_string: The `.rail` string.
            num_reasks: The max times to re-ask the LLM for invalid output.

        Returns:
            An instance of the `Guard` class.
        """
        output_schema, base_prompt, _ = read_rail(rail_string=rail_string)
        return cls(output_schema, base_prompt, num_reasks=num_reasks)

    def __call__(
        self,
        llm_api: Callable,
        prompt_params: Dict = None,
        num_reasks: int = 1,
        *args,
        **kwargs,
    ) -> Tuple[str, Dict]:
        """Outermost function that calls the LLM and validates the output.

        Args:
            llm_api: The LLM API to call (e.g. openai.Completion.create)
            prompt_params: The parameters to pass to the prompt.format() method.
            num_reasks: The max times to re-ask the LLM for invalid output.
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

            return self.ask_with_validation(prompt, llm_ask, num_reasks)

    def ask_with_validation(
        self, prompt: str, llm_ask: Callable, num_reasks: Optional[int]
    ) -> Tuple[str, Dict]:
        """Ask a question, and validate the output."""
        with start_action(action_type="ask_with_validation", prompt=prompt):
            guard_history = self.validation_inner_loop(
                prompt=prompt,
                llm_ask=llm_ask,
                reask_ctr=0,
                output_schema=self.output_schema,
                num_reasks=num_reasks,
            )

            return (
                guard_history.output,
                guard_history.validated_response,
            )

    def validation_inner_loop(
        self,
        reask_ctr: int,
        output_schema: OutputSchema,
        llm_ask: Optional[Callable] = None,
        prompt: Optional[str] = None,
        llm_output: Optional[str] = None,
        guard_history: GuardHistory = None,
        num_reasks: Optional[int] = None,
    ) -> GuardHistory:
        """Ask a question, and validate the output.

        Args:
            reask_ctr: The number of times the LLM has been reasked.
            output_schema: The output schema to validate against.
            llm_ask: The LLM API wrapper to call (e.g. wrapper openai.Completion.create)
            prompt: The prompt to send to the LLM. This or llm_output must be set.
            llm_output: The raw output from the LLM. This or prompt must be provided.
            guard_history: The history of the guard calls.

        Returns:
            The raw output from the LLM, the output as a dict, and the
            validated output.
        """

        if guard_history is None:
            guard_history = GuardHistory([])

        if num_reasks is None:
            num_reasks = self.num_reasks

        # If the prompt is not provided, then the output must be provided.
        if prompt is not None:
            assert llm_ask is not None
            assert llm_output is None
        elif llm_output is not None:
            assert prompt is None

        with start_action(
            action_type="validation_inner_loop", reask_ctr=reask_ctr
        ) as action:
            if llm_output is None:
                llm_output = llm_ask(prompt)
                action.log(message_type="info", prompt=prompt, output=llm_output)

            try:
                output_as_dict = json.loads(llm_output)
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
                prompt=prompt,
                output=llm_output,
                output_as_dict=output_as_dict,
                validated_response=validated_response,
                reasks=reasks,
            )

            guard_history = guard_history.push(gd_log)

            if len(reasks) and reask_ctr < num_reasks:
                if llm_ask is None:
                    # If the LLM API is None, then we can't re-ask the LLM.
                    self.guard_state = self.guard_state.push(guard_history)
                    return guard_history

                reask_json = reask_utils.prune_json_for_reasking(validated_response)
                reask_prompt, reask_schema = reask_utils.get_reask_prompt(
                    self.output_schema.parsed_rail, reasks, reask_json
                )

                return self.validation_inner_loop(
                    prompt=reask_prompt,
                    llm_ask=llm_ask,
                    reask_ctr=reask_ctr + 1,
                    output_schema=reask_schema,
                    guard_history=guard_history,
                    num_reasks=num_reasks,
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

        if check_refrain_in_dict(validated_response):
            logger.debug("Refrain detected.")
            validated_response = {}

        validated_response = filter_in_dict(validated_response)

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

    def parse(
        self,
        llm_output: str,
        llm_api: Callable = None,
        num_reasks: int = 1,
        *args,
        **kwargs,
    ) -> Dict:
        """Alternate flow to using Guard where the llm_output is known.

        Args:
            llm_output: The output from the LLM.
            llm_api: The LLM API to use to re-ask the LLM.
            num_reasks: The max times to re-ask the LLM for invalid output.

        Returns:
            The validated response.
        """

        llm_ask = None
        if llm_api is not None:
            llm_ask = get_llm_ask(llm_api, *args, **kwargs)

        guard_history = self.validation_inner_loop(
            reask_ctr=0,
            output_schema=self.output_schema,
            llm_ask=llm_ask,
            llm_output=llm_output,
            num_reasks=num_reasks,
        )

        validated_response = reask_utils.sub_reasks_with_fixed_values(
            guard_history.validated_response
        )

        return validated_response
