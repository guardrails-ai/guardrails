import json
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple

from eliot import start_action

from guardrails.llm_providers import PromptCallable
from guardrails.prompt import Instructions, Prompt
from guardrails.schema import InputSchema, OutputSchema
from guardrails.utils.logs_utils import GuardHistory, GuardLogs
from guardrails.utils.reask_utils import (
    ReAsk,
    gather_reasks,
    get_reask_prompt,
    prune_json_for_reasking,
    reask_json_as_dict,
    sub_reasks_with_fixed_values,
)


@dataclass
class Runner:
    """Runner class that calls an LLM API with a prompt, and performs input and
    output validation.

    This class will repeatedly call the API until the
    reask budget is exhausted, or the output is valid.

    Args:
        prompt: The prompt to use.
        api: The LLM API to call, which should return a string.
        input_schema: The input schema to use for validation.
        output_schema: The output schema to use for validation.
        num_reasks: The maximum number of times to reask the LLM in case of
            validation failure, defaults to 0.
        output: The output to use instead of calling the API, used in cases
            where the output is already known.
        guard_history: The guard history to use, defaults to an empty history.
    """

    instructions: Optional[Instructions]
    prompt: Prompt
    api: PromptCallable
    input_schema: InputSchema
    output_schema: OutputSchema
    num_reasks: int = 0
    output: str = None
    reask_prompt: Optional[Prompt] = None
    guard_history: GuardHistory = field(default_factory=lambda: GuardHistory([]))

    def _reset_guard_history(self):
        """Reset the guard history."""
        self.guard_history = GuardHistory([])

    def __post_init__(self):
        assert (self.prompt and self.api and not self.output) or (
            self.output and not self.prompt
        ), "Must provide either prompt and api or output."

    def __call__(self, prompt_params: Dict = None) -> GuardHistory:
        """Execute the runner by repeatedly calling step until the reask budget
        is exhausted.

        Args:
            prompt_params: Parameters to pass to the prompt in order to
                generate the prompt string.

        Returns:
            The guard history.
        """
        self._reset_guard_history()

        with start_action(
            action_type="run",
            instructions=self.instructions,
            prompt=self.prompt,
            api=self.api,
            input_schema=self.input_schema,
            output_schema=self.output_schema,
            num_reasks=self.num_reasks,
        ):
            instructions, prompt, input_schema, output_schema = (
                self.instructions,
                self.prompt,
                self.input_schema,
                self.output_schema,
            )
            for index in range(self.num_reasks + 1):
                # Run a single step.
                validated_output, reasks = self.step(
                    index=index,
                    api=self.api,
                    instructions=instructions,
                    prompt=prompt,
                    prompt_params=prompt_params,
                    input_schema=input_schema,
                    output_schema=output_schema,
                    output=self.output if index == 0 else None,
                )

                # Loop again?
                if not self.do_loop(index, reasks):
                    break
                # Get new prompt and output schema.
                prompt, output_schema = self.prepare_to_loop(
                    reasks,
                    validated_output,
                    output_schema,
                )

            return self.guard_history

    def step(
        self,
        index: int,
        api: Callable,
        instructions: Optional[Instructions],
        prompt: Prompt,
        prompt_params: Dict,
        input_schema: InputSchema,
        output_schema: OutputSchema,
        output: str = None,
    ):
        """Run a full step."""
        with start_action(
            action_type="step",
            index=index,
            instructions=instructions,
            prompt=prompt,
            prompt_params=prompt_params,
            input_schema=input_schema,
            output_schema=output_schema,
        ):
            # Prepare: run pre-processing, and input validation.
            if not output:
                instructions, prompt = self.prepare(
                    index, instructions, prompt, prompt_params, input_schema
                )
            else:
                instructions = None
                prompt = None

            # Call: run the API, and convert to dict.
            output, output_as_dict = self.call(index, instructions, prompt, api, output)

            # Validate: run output validation.
            validated_output = self.validate(index, output_as_dict, output_schema)

            # Introspect: inspect validated output for reasks.
            reasks = self.introspect(index, validated_output)

            # Replace reask values with fixed values if terminal step.
            if not self.do_loop(index, reasks):
                validated_output = sub_reasks_with_fixed_values(validated_output)

            # Log: step information.
            self.log(
                prompt=prompt,
                instructions=instructions,
                output=output,
                output_as_dict=output_as_dict,
                validated_output=validated_output,
                reasks=reasks,
            )

            return validated_output, reasks

    def prepare(
        self,
        index: int,
        instructions: Optional[Instructions],
        prompt: Prompt,
        prompt_params: Dict,
        input_schema: InputSchema,
    ) -> Prompt:
        """Prepare by running pre-processing and input validation."""
        with start_action(action_type="prepare", index=index) as action:
            if prompt_params is None:
                prompt_params = {}

            if input_schema:
                validated_prompt_params = input_schema.validate(prompt_params)
            else:
                validated_prompt_params = prompt_params

            if isinstance(prompt, str):
                prompt = Prompt(prompt)

            prompt = prompt.format(**validated_prompt_params)

            # TODO(shreya): should there be any difference to parsing params for prompt?
            if instructions is not None and isinstance(instructions, Instructions):
                instructions = instructions.format(**validated_prompt_params)

            action.log(
                message_type="info",
                instructions=instructions,
                prompt=prompt,
                prompt_params=prompt_params,
                validated_prompt_params=validated_prompt_params,
            )

        return instructions, prompt

    def post_process(self, output: str) -> str:
        """Post-process the raw output before parsing it.

        If the output is surrounded by triple backticks, remove them."""
        output = output.strip()
        if output.startswith("```"):
            output = output[3:]
            if output.startswith("json"):
                output = output[4:]
        if output.endswith("```"):
            output = output[:-3]
        return output

    def call(
        self,
        index: int,
        instructions: Optional[str],
        prompt: Prompt,
        api: Callable,
        output: str = None,
    ) -> Tuple[str, Optional[Dict]]:
        """Run a step.

        1. Query the LLM API,
        2. Convert the response string to a dict,
        3. Log the output
        """
        with start_action(action_type="call", index=index, prompt=prompt) as action:
            if prompt and instructions:
                output = api(prompt.source, instructions=instructions.source)
            elif prompt:
                output = api(prompt.source)

            # Post-process the output before loading it as JSON.
            output = self.post_process(output)

            error = None
            # Treat the output as a JSON string, and load it into a dict.
            try:
                output_as_dict = json.loads(output, strict=False)
            except json.decoder.JSONDecodeError as e:
                output_as_dict = None
                error = e

            action.log(
                message_type="info",
                output=output,
                output_as_dict=output_as_dict,
                error=error,
            )

            return output, output_as_dict

    def validate(
        self,
        index: int,
        output_as_dict: Dict,
        output_schema: OutputSchema,
    ):
        """Validate the output."""
        with start_action(action_type="validate", index=index) as action:
            validated_output = output_schema.validate(output_as_dict)
            action.log(
                message_type="info",
                validated_output=reask_json_as_dict(validated_output),
            )
            return validated_output

    def introspect(
        self,
        index: int,
        validated_output: Optional[Dict],
    ) -> List[ReAsk]:
        """Introspect the validated output."""
        with start_action(action_type="introspect", index=index) as action:
            if validated_output is None:
                return []
            reasks = gather_reasks(validated_output)
            action.log(
                message_type="info",
                reasks=[r.__dict__ for r in reasks],
            )
            return reasks

    def log(
        self,
        prompt: str,
        instructions: Optional[str],
        output: str,
        output_as_dict: Dict,
        validated_output: Dict,
        reasks: list,
    ) -> None:
        """Log the step."""
        self.guard_history = self.guard_history.push(
            GuardLogs(
                prompt=prompt,
                instructions=instructions,
                output=output,
                output_as_dict=output_as_dict,
                validated_output=validated_output,
                reasks=reasks,
            )
        )

    def do_loop(self, index: int, reasks: List[ReAsk]) -> bool:
        """Determine if we should loop again."""
        if reasks and index < self.num_reasks:
            return True
        return False

    def prepare_to_loop(
        self,
        reasks: list,
        validated_output: Optional[Dict],
        output_schema: OutputSchema,
    ) -> Tuple[str, OutputSchema]:
        """Prepare to loop again."""
        prompt, output_schema = get_reask_prompt(
            parsed_rail=output_schema.root,
            reasks=reasks,
            reask_json=prune_json_for_reasking(validated_output),
            reask_prompt_template=self.reask_prompt,
        )
        return prompt, OutputSchema(output_schema)
