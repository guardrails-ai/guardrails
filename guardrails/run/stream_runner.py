from typing import Any, Dict, Iterator, List, Optional, Tuple, Union, cast

from guardrails import validator_service
from guardrails.classes.history import Call, Inputs, Iteration, Outputs
from guardrails.classes.output_type import OT, OutputTypes
from guardrails.classes.validation_outcome import ValidationOutcome
from guardrails.llm_providers import (
    PromptCallableBase,
)
from guardrails.run.runner import Runner
from guardrails.hub_telemetry.hub_tracing import trace_stream
from guardrails.utils.parsing_utils import (
    coerce_types,
    parse_llm_output,
    prune_extra_keys,
)
from guardrails.actions.reask import ReAsk, SkeletonReAsk
from guardrails.constants import pass_status
from guardrails.telemetry import trace_stream_step


class StreamRunner(Runner):
    """Runner class that calls a streaming LLM API with a prompt.

    This class performs output validation when the output is a stream of
    chunks. Inherits from Runner class, as overall structure remains
    similar.
    """

    @trace_stream(name="/reasks", origin="StreamRunner.__call__")
    def __call__(
        self, call_log: Call, prompt_params: Optional[Dict] = {}
    ) -> Iterator[ValidationOutcome[OT]]:
        """Execute the StreamRunner.

        Args:
            prompt_params: Parameters to pass to the prompt in order to
                generate the prompt string.

        Returns:
            The Call log for this run.
        """

        prompt_params = prompt_params or {}

        (
            messages,
            output_schema,
        ) = (
            self.messages,
            self.output_schema,
        )

        return self.step(
            index=0,
            api=self.api,
            messages=messages,
            prompt_params=prompt_params,
            output_schema=output_schema,
            output=self.output,
            call_log=call_log,
        )

    @trace_stream(name="/step", origin="StreamRunner.step")
    @trace_stream_step
    def step(
        self,
        index: int,
        api: Optional[PromptCallableBase],
        messages: Optional[List[Dict]],
        prompt_params: Dict,
        output_schema: Dict[str, Any],
        call_log: Call,
        output: Optional[str] = None,
    ) -> Iterator[ValidationOutcome[OT]]:
        """Run a full step."""
        inputs = Inputs(
            llm_api=api,
            llm_output=output,
            messages=messages,
            prompt_params=prompt_params,
            num_reasks=self.num_reasks,
            metadata=self.metadata,
            full_schema_reask=self.full_schema_reask,
            stream=True,
        )
        outputs = Outputs()
        iteration = Iteration(
            call_id=call_log.id, index=index, inputs=inputs, outputs=outputs
        )
        call_log.iterations.push(iteration)

        # Prepare: run pre-processing, and input validation.
        if output is not None:
            messages = None
        else:
            messages = self.prepare(
                call_log,
                index,
                messages=messages,
                prompt_params=prompt_params,
                api=api,
            )

        iteration.inputs.messages = messages

        # Call: run the API that returns a generator wrapped in LLMResponse
        llm_response = self.call(messages, api, output)

        iteration.outputs.llm_response_info = llm_response

        # Get the stream (generator) from the LLMResponse
        stream = llm_response.stream_output
        if stream is None:
            raise ValueError(
                "No stream was returned from the API. Please check that "
                "the API is returning a generator."
            )

        parsed_fragment, validated_fragment, valid_op = "", None, None
        verified = set()
        validation_response = ""
        fragment = ""
        # Loop over the stream
        # and construct "fragments" of concatenated chunks
        # for now, handle string and json schema differently
        if self.output_type == OutputTypes.STRING:

            def prepare_chunk_generator(stream) -> Iterator[Tuple[Any, bool]]:
                for chunk in stream:
                    chunk_text = self.get_chunk_text(chunk, api)
                    nonlocal fragment
                    fragment += chunk_text
                    finished = self.is_last_chunk(chunk, api)
                    # 2. Parse the chunk
                    parsed_chunk, move_to_next = self.parse(
                        chunk_text, output_schema, verified=verified
                    )
                    nonlocal parsed_fragment
                    # ignore types because output schema guarantees a string
                    parsed_fragment += parsed_chunk  # type: ignore
                    if move_to_next:
                        # Continue to next chunk
                        continue
                    yield parsed_chunk, finished

            prepped_stream = prepare_chunk_generator(stream)
            gen = validator_service.validate_stream(
                prepped_stream,
                self.metadata,
                self.validation_map,
                iteration,
                self._disable_tracer,
                "$",
                validate_subschema=True,
            )

            for res in gen:
                chunk = res.chunk
                original_text = res.original_text
                if isinstance(chunk, SkeletonReAsk):
                    raise ValueError(
                        "Received fragment schema is an invalid sub-schema "
                        "of the expected output JSON schema."
                    )

                # 4. Introspect: inspect the validated fragment for reasks
                reasks, valid_op = self.introspect(chunk)
                if reasks:
                    raise ValueError(
                        "Reasks are not yet supported with streaming. Please "
                        "remove reasks from schema or disable streaming."
                    )
                # 5. Convert validated fragment to a pretty JSON string
                validation_response += cast(str, chunk)
                passed = call_log.status == pass_status
                yield ValidationOutcome(
                    call_id=call_log.id,  # type: ignore
                    #  The chunk or the whole output?
                    raw_llm_output=original_text,
                    validated_output=chunk,
                    validation_passed=passed,
                )

        # handle non string schema
        else:
            for chunk in stream:
                # 1. Get the text from the chunk and append to fragment
                chunk_text = self.get_chunk_text(chunk, api)
                fragment += chunk_text

                # 2. Parse the fragment
                parsed_fragment, move_to_next = self.parse(
                    fragment, output_schema, verified=verified
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
                reasks, valid_op = self.introspect(validated_fragment)
                if reasks:
                    raise ValueError(
                        "Reasks are not yet supported with streaming. Please "
                        "remove reasks from schema or disable streaming."
                    )

                if self.output_type == OutputTypes.LIST:
                    validation_response = cast(list, validated_fragment)
                else:
                    validation_response = cast(dict, validated_fragment)
                # 5. Convert validated fragment to a pretty JSON string
                yield ValidationOutcome(
                    call_id=call_log.id,  # type: ignore
                    raw_llm_output=fragment,
                    validated_output=validated_fragment,
                    validation_passed=validated_fragment is not None,
                )

        # # Finally, add to logs
        iteration.outputs.raw_output = fragment
        iteration.outputs.parsed_output = parsed_fragment or fragment  # type: ignore
        iteration.outputs.validation_response = validation_response
        iteration.outputs.guarded_output = valid_op

    def is_last_chunk(self, chunk: Any, api: Union[PromptCallableBase, None]) -> bool:
        """Detect if chunk is final chunk."""
        try:
            finished = chunk.choices[0].finish_reason
            return finished is not None
        except (AttributeError, TypeError):
            return False

    def get_chunk_text(self, chunk: Any, api: Union[PromptCallableBase, None]) -> str:
        """Get the text from a chunk."""
        chunk_text = ""
        try:
            finished = chunk.choices[0].finish_reason
            content = chunk.choices[0].delta.content
            if not finished and content:
                chunk_text = content
        except Exception:
            try:
                finished = chunk.choices[0].finish_reason
                content = chunk.choices[0].text
                if not finished and content:
                    chunk_text = content
            except Exception:
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
        self, output: str, output_schema: Dict[str, Any], *, verified: set, **kwargs
    ):
        """Parse the output."""
        parsed_output, error = parse_llm_output(
            output, self.output_type, stream=True, verified=verified
        )

        if parsed_output and not error and not isinstance(parsed_output, ReAsk):
            parsed_output = prune_extra_keys(parsed_output, output_schema)
            parsed_output = coerce_types(parsed_output, output_schema)

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
