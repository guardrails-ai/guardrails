from typing import Any, Dict, Generator, List, Optional, Union

from guardrails.classes.history import Call, Inputs, Iteration, Outputs
from guardrails.classes.output_type import OT
from guardrails.classes.validation_outcome import ValidationOutcome
from guardrails.datatypes import verify_metadata_requirements
from guardrails.llm_providers import (
    LiteLLMCallable,
    OpenAICallable,
    OpenAIChatCallable,
    PromptCallableBase,
)
from guardrails.prompt import Instructions, Prompt
from guardrails.run.runner import Runner
from guardrails.schema import Schema, StringSchema
from guardrails.utils.openai_utils import OPENAI_VERSION
from guardrails.utils.reask_utils import SkeletonReAsk


class StreamRunner(Runner):
    """Runner class that calls a streaming LLM API with a prompt.

    This class performs output validation when the output is a stream of
    chunks. Inherits from Runner class, as overall structure remains
    similar.
    """

    def __call__(
        self, call_log: Call, prompt_params: Optional[Dict] = None
    ) -> Generator[ValidationOutcome[OT], None, None]:
        """Execute the StreamRunner.

        Args:
            prompt_params: Parameters to pass to the prompt in order to
                generate the prompt string.

        Returns:
            The Call log for this run.
        """
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

        (
            instructions,
            prompt,
            msg_history,
            prompt_schema,
            instructions_schema,
            msg_history_schema,
            output_schema,
        ) = (
            self.instructions,
            self.prompt,
            self.msg_history,
            self.prompt_schema,
            self.instructions_schema,
            self.msg_history_schema,
            self.output_schema,
        )

        return self.step(
            index=0,
            api=self.api,
            instructions=instructions,
            prompt=prompt,
            msg_history=msg_history,
            prompt_params=prompt_params,
            prompt_schema=prompt_schema,
            instructions_schema=instructions_schema,
            msg_history_schema=msg_history_schema,
            output_schema=output_schema,
            output=self.output,
            call_log=call_log,
        )

    def step(
        self,
        index: int,
        api: Optional[PromptCallableBase],
        instructions: Optional[Instructions],
        prompt: Optional[Prompt],
        msg_history: Optional[List[Dict]],
        prompt_params: Dict,
        prompt_schema: Optional[StringSchema],
        instructions_schema: Optional[StringSchema],
        msg_history_schema: Optional[StringSchema],
        output_schema: Schema,
        call_log: Call,
        output: Optional[str] = None,
    ) -> Generator[ValidationOutcome[OT], None, None]:
        """Run a full step."""
        inputs = Inputs(
            llm_api=api,
            llm_output=output,
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
        call_log.iterations.push(iteration)

        # Prepare: run pre-processing, and input validation.
        if output:
            instructions = None
            prompt = None
            msg_history = None
        else:
            instructions, prompt, msg_history = self.prepare(
                call_log,
                index,
                instructions,
                prompt,
                msg_history,
                prompt_params,
                api,
                prompt_schema,
                instructions_schema,
                msg_history_schema,
                output_schema,
            )

        iteration.inputs.prompt = prompt
        iteration.inputs.instructions = instructions
        iteration.inputs.msg_history = msg_history

        # Call: run the API that returns a generator wrapped in LLMResponse
        llm_response = self.call(index, instructions, prompt, msg_history, api, output)

        # Get the stream (generator) from the LLMResponse
        stream = llm_response.stream_output
        if stream is None:
            raise ValueError(
                "No stream was returned from the API. Please check that "
                "the API is returning a generator."
            )

        fragment = ""
        parsed_fragment, validated_fragment, valid_op = None, None, None
        verified = set()
        # Loop over the stream
        # and construct "fragments" of concatenated chunks
        for chunk in stream:
            # 1. Get the text from the chunk and append to fragment
            chunk_text = self.get_chunk_text(chunk, api)
            fragment += chunk_text

            # 2. Parse the fragment
            parsed_fragment, move_to_next = self.parse(
                index, fragment, output_schema, verified
            )
            if move_to_next:
                # Continue to next chunk
                continue

            # 3. Run output validation
            validated_fragment = self.validate(
                iteration,
                index,
                parsed_fragment,
                output_schema,
                validate_subschema=True,
            )
            if isinstance(validated_fragment, SkeletonReAsk):
                raise ValueError(
                    "Received fragment schema is an invalid sub-schema "
                    "of the expected output JSON schema."
                )

            # 4. Introspect: inspect the validated fragment for reasks
            reasks, valid_op = self.introspect(index, validated_fragment, output_schema)
            if reasks:
                raise ValueError(
                    "Reasks are not yet supported with streaming. Please "
                    "remove reasks from schema or disable streaming."
                )

            # 5. Convert validated fragment to a pretty JSON string
            yield ValidationOutcome(
                raw_llm_output=fragment,
                validated_output=validated_fragment,
                validation_passed=validated_fragment is not None,
            )

        # Finally, add to logs
        iteration.outputs.raw_output = fragment
        iteration.outputs.parsed_output = parsed_fragment
        iteration.outputs.validation_response = validated_fragment
        iteration.outputs.guarded_output = valid_op

    def get_chunk_text(self, chunk: Any, api: Union[PromptCallableBase, None]) -> str:
        """Get the text from a chunk."""
        chunk_text = ""
        if isinstance(api, OpenAICallable):
            if OPENAI_VERSION.startswith("0"):
                finished = chunk["choices"][0]["finish_reason"]
                if "text" in chunk["choices"][0]:
                    content = chunk["choices"][0]["text"]
                    if not finished and content:
                        chunk_text = content
            else:
                finished = chunk.choices[0].finish_reason
                content = chunk.choices[0].text
                if not finished and content:
                    chunk_text = content
        elif isinstance(api, OpenAIChatCallable):
            if OPENAI_VERSION.startswith("0"):
                finished = chunk["choices"][0]["finish_reason"]
                if "content" in chunk["choices"][0]["delta"]:
                    content = chunk["choices"][0]["delta"]["content"]
                    if not finished and content:
                        chunk_text = content
            else:
                finished = chunk.choices[0].finish_reason
                content = chunk.choices[0].delta.content
                if not finished and content:
                    chunk_text = content
        elif isinstance(api, LiteLLMCallable):
            finished = chunk.choices[0].finish_reason
            content = chunk.choices[0].delta.content
            if not finished and content:
                chunk_text = content
        else:
            try:
                chunk_text = chunk
            except Exception as e:
                raise ValueError(
                    f"Error getting chunk from stream: {e}. "
                    "Non-OpenAI API callables expected to return "
                    "a generator of strings."
                ) from e
        return chunk_text

    def parse(
        self,
        index: int,
        output: str,
        output_schema: Schema,
        verified: set,
    ):
        """Parse the output."""
        parsed_output, error = output_schema.parse(
            output, stream=True, verified=verified
        )

        # Error can be either of
        # (True/False/None/ValueError/string representing error)
        if error:
            # If parsing error is a string,
            # it is an error from output_schema.parse_fragment()
            if isinstance(error, str):
                raise ValueError("Unable to parse output: " + error)
        # Else if either of
        # (None/True/False/ValueError), return parsed_output and error

        return parsed_output, error
