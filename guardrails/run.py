import copy
import logging
from typing import Any, Dict, List, Optional, Sequence, Tuple, Type, Union

from eliot import add_destinations, start_action
from pydantic import BaseModel

from guardrails.classes.generic import Stack
from guardrails.classes.history import Call, Inputs, Iteration, Outputs
from guardrails.datatypes import verify_metadata_requirements
from guardrails.llm_providers import AsyncPromptCallableBase, PromptCallableBase
from guardrails.prompt import Instructions, Prompt
from guardrails.schema import Schema
from guardrails.utils.history_utils import merge_valid_output
from guardrails.utils.llm_response import LLMResponse
from guardrails.utils.logs_utils import merge_reask_output
from guardrails.utils.reask_utils import (
    FieldReAsk,
    NonParseableReAsk,
    ReAsk,
    SkeletonReAsk,
    reasks_to_dict,
    sub_reasks_with_fixed_values,
)

logger = logging.getLogger(__name__)
actions_logger = logging.getLogger(f"{__name__}.actions")
add_destinations(actions_logger.debug)


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
        current_call: The call history to use, defaults to an empty Call.
    """

    def __init__(
        self,
        output_schema: Schema,
        history: Stack[Call],
        num_reasks: int,
        prompt: Optional[Union[str, Prompt]] = None,
        instructions: Optional[Union[str, Instructions]] = None,
        msg_history: Optional[List[Dict]] = None,
        api: Optional[PromptCallableBase] = None,
        input_schema: Optional[Schema] = None,
        metadata: Optional[Dict[str, Any]] = None,
        output: Optional[str] = None,
        current_call: Optional[Call] = None,
        base_model: Optional[Type[BaseModel]] = None,
        full_schema_reask: bool = False,
    ):
        if prompt:
            assert api, "Must provide an API if a prompt is provided."
            assert not output, "Cannot provide both a prompt and output."

        if isinstance(prompt, str):
            self.prompt = Prompt(prompt, output_schema=output_schema.transpile())
        else:
            self.prompt = prompt

        if isinstance(instructions, str):
            self.instructions = Instructions(
                instructions, output_schema=output_schema.transpile()
            )
        else:
            self.instructions = instructions

        if msg_history:
            msg_history = copy.deepcopy(msg_history)
            msg_history_copy = []
            for msg in msg_history:
                msg["content"] = Prompt(
                    msg["content"], output_schema=output_schema.transpile()
                )
                msg_history_copy.append(msg)
            self.msg_history = msg_history_copy
        else:
            self.msg_history = None

        self.api = api
        self.input_schema = input_schema
        self.output_schema = output_schema
        self.history = history
        self.num_reasks = num_reasks
        self.metadata = metadata or {}
        self.output = output
        self.current_call = current_call or Call()
        self.base_model = base_model
        self.full_schema_reask = full_schema_reask

    def _add_call_to_history(self):
        """Reset the guard history."""
        self.current_call = Call()
        self.history.push(self.current_call)

    def __call__(
        self, prompt_params: Optional[Dict] = None
    ) -> Tuple[Call, Optional[str]]:
        """Execute the runner by repeatedly calling step until the reask budget
        is exhausted.

        Args:
            prompt_params: Parameters to pass to the prompt in order to
                generate the prompt string.

        Returns:
            The guard history.
        """
        error_message = None
        try:
            if prompt_params is None:
                prompt_params = {}

            # check if validator requirements are fulfilled
            missing_keys = verify_metadata_requirements(
                self.metadata, self.output_schema.root_datatype
            )
            if missing_keys:
                raise ValueError(
                    f"Missing required metadata keys: {', '.join(missing_keys)}"
                )

            self._add_call_to_history()

            # Figure out if we need to include instructions in the prompt.
            include_instructions = not (
                self.instructions is None and self.msg_history is None
            )

            with start_action(
                action_type="run",
                instructions=self.instructions,
                prompt=self.prompt,
                api=self.api,
                input_schema=self.input_schema,
                output_schema=self.output_schema,
                num_reasks=self.num_reasks,
                metadata=self.metadata,
            ):
                instructions, prompt, msg_history, input_schema, output_schema = (
                    self.instructions,
                    self.prompt,
                    self.msg_history,
                    self.input_schema,
                    self.output_schema,
                )
                for index in range(self.num_reasks + 1):
                    # Run a single step.
                    iteration = self.step(
                        index=index,
                        api=self.api,
                        instructions=instructions,
                        prompt=prompt,
                        msg_history=msg_history,
                        prompt_params=prompt_params,
                        input_schema=input_schema,
                        output_schema=output_schema,
                        output=self.output if index == 0 else None,
                    )
                    # Loop again?
                    if not self.do_loop(index, iteration.reasks):
                        break

                    # Get merged validation output for prompt
                    print("iteration.validation_output: ", iteration.validation_output)
                    validation_output = iteration.validation_output
                    if (
                        self.current_call.iterations.length > 1
                        and not self.full_schema_reask
                        and not isinstance(validation_output, SkeletonReAsk)
                    ):
                        print("Calling merge_reask_output...")
                        validation_output = merge_reask_output(
                            self.current_call.iterations.at(
                                index - 1
                            ).validation_output,
                            self.current_call.iterations.last.validation_output,
                        )
                        print(
                            "validation_output AFTER merge_reask_output: ",
                            validation_output,
                        )

                    # Get new prompt and output schema.
                    (
                        prompt,
                        instructions,
                        output_schema,
                        msg_history,
                    ) = self.prepare_to_loop(
                        iteration.reasks,
                        validation_output,
                        output_schema,
                        prompt_params=prompt_params,
                        include_instructions=include_instructions,
                    )

        except Exception as e:
            error_message = str(e)
        return self.current_call, error_message

    def step(
        self,
        index: int,
        api: Optional[PromptCallableBase],
        instructions: Optional[Instructions],
        prompt: Optional[Prompt],
        msg_history: Optional[List[Dict]],
        prompt_params: Dict,
        input_schema: Optional[Schema],
        output_schema: Schema,
        output: Optional[str] = None,
    ) -> Iteration:
        inputs = Inputs(
            llm_api=api,
            llm_response=output,
            instructions=instructions,
            prompt=prompt,
            msg_history=msg_history,
            prompt_params=prompt_params,
            num_reasks=self.num_reasks,
            metadata=self.metadata,
            full_schema_reask=self.full_schema_reask,
        )
        outputs = Outputs()
        iteration = Iteration(inputs=inputs, outputs=outputs)
        self.current_call.iterations.push(iteration)

        print("Running step number ", index)
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
            if output:
                instructions = None
                prompt = None
                msg_history = None
            else:
                instructions, prompt, msg_history = self.prepare(
                    index,
                    instructions,
                    prompt,
                    msg_history,
                    prompt_params,
                    api,
                    input_schema,
                    output_schema,
                )

            iteration.inputs.prompt = prompt
            iteration.inputs.instructions = instructions
            iteration.inputs.msg_history = msg_history

            # Call: run the API.
            print("Calling the llm...")
            llm_response = self.call(
                index, instructions, prompt, msg_history, api, output
            )
            print("Llm response received!")

            iteration.outputs.llm_response_info = llm_response
            raw_output = llm_response.output

            # Parse: parse the output.
            print("Parsing...")
            parsed_output, parsing_error = self.parse(index, raw_output, output_schema)
            print("Parsing complete!")

            iteration.outputs.parsed_output = parsed_output
            iteration.outputs.error = parsing_error

            # Validate: run output validation.
            validated_output = None
            if parsing_error and isinstance(parsed_output, NonParseableReAsk):
                reasks = self.introspect(index, parsed_output, output_schema)
            else:
                # Validate: run output validation.
                print("Validating...")
                validated_output = self.validate(
                    iteration, index, parsed_output, output_schema
                )
                print("Validation complete!")
                iteration.outputs.validation_output = validated_output

                # Introspect: inspect validated output for reasks.
                reasks, valid_output = self.introspect(
                    index, validated_output, output_schema
                )
                iteration.outputs.validated_output = valid_output

            iteration.outputs.reasks = reasks

            # Replace reask values with fixed values if terminal step.
            if not self.do_loop(index, reasks):
                print("Merging final output...")
                final_valid_output = merge_valid_output(self.current_call)
                final_output = sub_reasks_with_fixed_values(validated_output)
                # TODO: Pass in a return type from Guard
                if (
                    not isinstance(final_valid_output, str)
                    and not isinstance(final_output, str)
                    and not isinstance(final_output, ReAsk)
                ):
                    final_valid_output = {**final_output, **final_valid_output}
                print("Merge complete!")
                iteration.outputs.validated_output = final_valid_output

            print("Returning...")
            return iteration

    def prepare(
        self,
        index: int,
        instructions: Optional[Instructions],
        prompt: Optional[Prompt],
        msg_history: Optional[List[Dict]],
        prompt_params: Dict,
        api: Optional[Union[PromptCallableBase, AsyncPromptCallableBase]],
        input_schema: Optional[Schema],
        output_schema: Schema,
    ) -> Tuple[Optional[Instructions], Optional[Prompt], Optional[List[Dict]]]:
        """Prepare by running pre-processing and input validation.

        Returns:
            The instructions, prompt, and message history.
        """
        with start_action(action_type="prepare", index=index) as action:
            if api is None:
                raise ValueError("API must be provided.")

            if prompt_params is None:
                prompt_params = {}

            if msg_history:
                msg_history = copy.deepcopy(msg_history)
                # Format any variables in the message history with the prompt params.
                for msg in msg_history:
                    msg["content"] = msg["content"].format(**prompt_params)

                prompt, instructions = None, None
            elif prompt is not None:
                if isinstance(prompt, str):
                    prompt = Prompt(prompt)

                prompt = prompt.format(**prompt_params)

                # TODO(shreya): should there be any difference
                #  to parsing params for prompt?
                if instructions is not None and isinstance(instructions, Instructions):
                    instructions = instructions.format(**prompt_params)

                instructions, prompt = output_schema.preprocess_prompt(
                    api, instructions, prompt
                )
            else:
                raise ValueError("Prompt or message history must be provided.")

            action.log(
                message_type="info",
                instructions=instructions,
                prompt=prompt,
                prompt_params=prompt_params,
                validated_prompt_params=prompt_params,
            )

        return instructions, prompt, msg_history

    def call(
        self,
        index: int,
        instructions: Optional[Instructions],
        prompt: Optional[Prompt],
        msg_history: Optional[List[Dict[str, str]]],
        api: Optional[PromptCallableBase],
        output: Optional[str] = None,
    ) -> LLMResponse:
        """Run a step.

        1. Query the LLM API,
        2. Convert the response string to a dict,
        3. Log the output
        """

        with start_action(action_type="call", index=index, prompt=prompt) as action:
            if output is not None:
                llm_response = LLMResponse(
                    output=output,
                )
            elif api is None:
                raise ValueError("API or output must be provided.")
            elif msg_history:
                try:
                    llm_response = api(
                        msg_history=msg_history_source(msg_history),
                        base_model=self.base_model,
                    )
                except Exception:
                    # If the API call fails, try calling again without the base model.
                    llm_response = api(msg_history=msg_history_source(msg_history))
            elif prompt and instructions:
                try:
                    llm_response = api(
                        prompt.source,
                        instructions=instructions.source,
                        base_model=self.base_model,
                    )
                except Exception:
                    llm_response = api(prompt.source, instructions=instructions.source)
            elif prompt:
                try:
                    llm_response = api(prompt.source, base_model=self.base_model)
                except Exception:
                    llm_response = api(prompt.source)
            else:
                raise ValueError("Prompt or message history must be provided.")

            action.log(
                message_type="info",
                output=llm_response,
            )

            return llm_response

    def parse(
        self,
        index: int,
        output: str,
        output_schema: Schema,
    ):
        with start_action(action_type="parse", index=index) as action:
            parsed_output, error = output_schema.parse(output)

            action.log(
                message_type="info",
                parsed_output=parsed_output,
                error=error,
            )

            return parsed_output, error

    def validate(
        self,
        iteration: Iteration,
        index: int,
        parsed_output: Any,
        output_schema: Schema,
    ):
        """Validate the output."""
        with start_action(action_type="validate", index=index) as action:
            validated_output = output_schema.validate(
                iteration, parsed_output, self.metadata
            )

            action.log(
                message_type="info",
                validated_output=reasks_to_dict(validated_output),
            )

            return validated_output

    def introspect(
        self,
        index: int,
        validated_output: Any,
        output_schema: Schema,
    ) -> Tuple[List[FieldReAsk], Union[str, Dict]]:
        """Introspect the validated output."""
        output = copy.deepcopy(validated_output)
        with start_action(action_type="introspect", index=index) as action:
            if output is None:
                return []
            reasks, valid_output = output_schema.introspect(output)

            action.log(
                message_type="info",
                reasks=[r.__dict__ for r in reasks],
            )

            return reasks, valid_output

    def do_loop(self, index: int, reasks: Sequence[ReAsk]) -> bool:
        """Determine if we should loop again."""
        if reasks and index < self.num_reasks:
            return True
        return False

    def prepare_to_loop(
        self,
        reasks: list,
        validated_output: Optional[Union[Dict, ReAsk]],
        output_schema: Schema,
        prompt_params: Dict,
        include_instructions: bool = False,
    ) -> Tuple[Prompt, Optional[Instructions], Schema, Optional[List[Dict]]]:
        """Prepare to loop again."""
        print(" !!!!!!!!!!!! START Runner.prepare_to_loop !!!!!!!!!!!!")
        print("validated_output: ", validated_output)
        print("type(validated_output): ", type(validated_output))
        print(" !!!!!!!!!!!! END Runner.prepare_to_loop !!!!!!!!!!!!")

        output_schema, prompt, instructions = output_schema.get_reask_setup(
            reasks=reasks,
            original_response=validated_output,
            use_full_schema=self.full_schema_reask,
            prompt_params=prompt_params,
        )
        if not include_instructions:
            instructions = None
        msg_history = None  # clear msg history for reasking
        return prompt, instructions, output_schema, msg_history


class AsyncRunner(Runner):
    def __init__(
        self,
        output_schema: Schema,
        history: Stack[Call],
        num_reasks: int,
        prompt: Optional[Union[str, Prompt]] = None,
        instructions: Optional[Union[str, Instructions]] = None,
        msg_history: Optional[List[Dict]] = None,
        api: Optional[AsyncPromptCallableBase] = None,
        input_schema: Optional[Schema] = None,
        metadata: Optional[Dict[str, Any]] = None,
        output: Optional[str] = None,
        current_call: Optional[Call] = None,
        base_model: Optional[Type[BaseModel]] = None,
        full_schema_reask: bool = False,
    ):
        super().__init__(
            output_schema=output_schema,
            history=history,
            num_reasks=num_reasks,
            prompt=prompt,
            instructions=instructions,
            msg_history=msg_history,
            api=api,
            input_schema=input_schema,
            metadata=metadata,
            output=output,
            current_call=current_call,
            base_model=base_model,
            full_schema_reask=full_schema_reask,
        )
        self.api: Optional[AsyncPromptCallableBase] = api

    async def async_run(
        self, prompt_params: Optional[Dict] = None
    ) -> Tuple[Call, Optional[str]]:
        """Execute the runner by repeatedly calling step until the reask budget
        is exhausted.

        Args:
            prompt_params: Parameters to pass to the prompt in order to
                generate the prompt string.

        Returns:
            The guard history.
        """
        error_message = None
        try:
            if prompt_params is None:
                prompt_params = {}
            self._add_call_to_history()

            # check if validator requirements are fulfilled
            missing_keys = verify_metadata_requirements(
                self.metadata, self.output_schema.root_datatype
            )
            if missing_keys:
                raise ValueError(
                    f"Missing required metadata keys: {', '.join(missing_keys)}"
                )

            with start_action(
                action_type="run",
                instructions=self.instructions,
                prompt=self.prompt,
                api=self.api,
                input_schema=self.input_schema,
                output_schema=self.output_schema,
                num_reasks=self.num_reasks,
                metadata=self.metadata,
            ):
                instructions, prompt, msg_history, input_schema, output_schema = (
                    self.instructions,
                    self.prompt,
                    self.msg_history,
                    self.input_schema,
                    self.output_schema,
                )
                for index in range(self.num_reasks + 1):
                    # Run a single step.
                    iteration = await self.async_step(
                        index=index,
                        api=self.api,
                        instructions=instructions,
                        prompt=prompt,
                        msg_history=msg_history,
                        prompt_params=prompt_params,
                        input_schema=input_schema,
                        output_schema=output_schema,
                        output=self.output if index == 0 else None,
                    )
                    self.current_call.iterations.push(iteration)

                    # Loop again?
                    if not self.do_loop(index, iteration.reasks):
                        break

                    # Get merged validation output for prompt
                    print("iteration.validation_output: ", iteration.validation_output)
                    validation_output = iteration.validation_output
                    if (
                        self.current_call.iterations.length > 1
                        and not self.full_schema_reask
                        and not isinstance(validation_output, SkeletonReAsk)
                    ):
                        print("Calling merge_reask_output...")
                        validation_output = merge_reask_output(
                            self.current_call.iterations.at(index - 1),
                            self.current_call.iterations.last,
                        )

                    # Get new prompt and output schema.
                    (
                        prompt,
                        instructions,
                        output_schema,
                        msg_history,
                    ) = self.prepare_to_loop(
                        iteration.reasks,
                        validation_output,
                        output_schema,
                        prompt_params=prompt_params,
                    )
        except Exception as e:
            error_message = str(e)

        return self.current_call, error_message

    async def async_step(
        self,
        index: int,
        api: Optional[AsyncPromptCallableBase],
        instructions: Optional[Instructions],
        prompt: Optional[Prompt],
        msg_history: Optional[List[Dict]],
        prompt_params: Dict,
        input_schema: Optional[Schema],
        output_schema: Schema,
        output: Optional[str] = None,
    ) -> Iteration:
        print("Running step number ", index)
        inputs = Inputs(
            llm_api=api,
            llm_response=output,
            instructions=instructions,
            prompt=prompt,
            msg_history=msg_history,
            prompt_params=prompt_params,
            num_reasks=self.num_reasks,
            metadata=self.metadata,
            full_schema_reask=self.full_schema_reask,
        )
        outputs = Outputs()
        iteration = Iteration(inputs=inputs, outputs=outputs)
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
            if output:
                instructions = None
                prompt = None
                msg_history = None
            else:
                instructions, prompt, msg_history = self.prepare(
                    index,
                    instructions,
                    prompt,
                    msg_history,
                    prompt_params,
                    api,
                    input_schema,
                    output_schema,
                )

            iteration.inputs.prompt = prompt
            iteration.inputs.instructions = instructions
            iteration.inputs.msg_history = msg_history

            # Call: run the API.
            print("Calling llm...")
            llm_response = await self.async_call(
                index, instructions, prompt, msg_history, api, output
            )
            print("Received response from llm!")

            iteration.outputs.llm_response_info = llm_response
            output = llm_response.output

            # Parse: parse the output.
            print("Parsing...")
            parsed_output, parsing_error = self.parse(index, output, output_schema)
            print("Parsing complete!")

            iteration.outputs.parsed_output = parsed_output

            validated_output = None
            if parsing_error and isinstance(parsed_output, NonParseableReAsk):
                reasks = self.introspect(index, parsed_output, output_schema)
            else:
                # Validate: run output validation.
                print("Validating...")
                validated_output = await self.async_validate(
                    iteration, index, parsed_output, output_schema
                )
                print("Validation complete!")
                iteration.outputs.validation_output = validated_output

                # Introspect: inspect validated output for reasks.
                reasks, valid_output = self.introspect(
                    index, validated_output, output_schema
                )
                iteration.outputs.validated_output = valid_output

            iteration.outputs.reasks = reasks

            # Replace reask values with fixed values if terminal step.
            if not self.do_loop(index, reasks):
                print("Merging final output...")
                final_valid_output = merge_valid_output(self.current_call)
                final_output = sub_reasks_with_fixed_values(validated_output)
                # TODO: Pass in a return type from Guard
                if (
                    not isinstance(final_valid_output, str)
                    and not isinstance(final_output, str)
                    and not isinstance(final_output, ReAsk)
                ):
                    final_valid_output = {**final_output, **final_valid_output}
                print("Merge complete!")
                iteration.outputs.validated_output = final_valid_output

            print("Returning...")
            return iteration

    async def async_call(
        self,
        index: int,
        instructions: Optional[Instructions],
        prompt: Optional[Prompt],
        msg_history: Optional[List[Dict]],
        api: Optional[AsyncPromptCallableBase],
        output: Optional[str] = None,
    ) -> LLMResponse:
        """Run a step.

        1. Query the LLM API,
        2. Convert the response string to a dict,
        3. Log the output
        """
        with start_action(action_type="call", index=index, prompt=prompt) as action:
            if output is not None:
                llm_response = LLMResponse(
                    output=output,
                )
            elif api is None:
                raise ValueError("Either API or output must be provided.")
            elif msg_history:
                try:
                    llm_response = await api(
                        msg_history=msg_history_source(msg_history),
                        base_model=self.base_model,
                    )
                except Exception:
                    # If the API call fails, try calling again without the base model.
                    llm_response = await api(
                        msg_history=msg_history_source(msg_history)
                    )
            elif prompt and instructions:
                try:
                    llm_response = await api(
                        prompt.source,
                        instructions=instructions.source,
                        base_model=self.base_model,
                    )
                except Exception:
                    llm_response = await api(
                        prompt.source, instructions=instructions.source
                    )
            elif prompt:
                try:
                    llm_response = await api(prompt.source, base_model=self.base_model)
                except Exception:
                    llm_response = await api(prompt.source)
            else:
                raise ValueError("Output, prompt or message history must be provided.")

            action.log(
                message_type="info",
                output=llm_response,
            )

            return llm_response

    async def async_validate(
        self,
        iteration: Iteration,
        index: int,
        parsed_output: Any,
        output_schema: Schema,
    ):
        """Validate the output."""
        with start_action(action_type="validate", index=index) as action:
            validated_output = await output_schema.async_validate(
                iteration, parsed_output, self.metadata
            )

            action.log(
                message_type="info",
                validated_output=reasks_to_dict(validated_output),
            )

            return validated_output


def msg_history_source(msg_history) -> List[Dict[str, str]]:
    msg_history_copy = copy.deepcopy(msg_history)
    for msg in msg_history_copy:
        msg["content"] = msg["content"].source
    return msg_history_copy
