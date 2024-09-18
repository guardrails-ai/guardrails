from typing import Callable, Dict, Iterable, List, Optional, Union, cast
import warnings
from typing_extensions import deprecated

from guardrails.classes.execution.guard_execution_options import GuardExecutionOptions
from guardrails.classes.output_type import OT, OutputTypes
from guardrails.classes.validation_outcome import ValidationOutcome

from guardrails import Guard
from nemoguardrails import LLMRails

from guardrails.formatters import get_formatter
from guardrails.formatters.base_formatter import BaseFormatter
from guardrails.schema.pydantic_schema import pydantic_model_to_schema
from guardrails.types.pydantic import ModelOrListOfModels

from guardrails.stores.context import (
    Tracer
)


class NemoguardrailsGuard(Guard):
    def __init__(
        self,
        nemorails: LLMRails,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self._nemorails = nemorails

    def __call__(
        self, llm_api: Optional[Callable] = None, generate_kwargs: Optional[Dict] = None, *args, **kwargs
    ) -> Union[ValidationOutcome[OT], Iterable[ValidationOutcome[OT]]]:
        # peel llm_api off of kwargs
        llm_api = kwargs.pop("llm_api", None)

        # if llm_api is defined, throw an error
        if llm_api is not None:
            raise ValueError(
                """llm_api should not be passed to a NemoguardrailsGuard object.
                The Nemoguardrails LLMRails object passed in will be used as the LLM."""
            )

        # peel off messages from kwargs
        messages = kwargs.get("messages", None)

        # if messages is not defined, throw an error
        if messages is None:
            raise ValueError(
                """messages should be passed to a NemoguardrailsGuard object.
 The messages to be passed to the LLM should be passed in as a list of
 dictionaries, where each dictionary has a 'role' key and a 'content' key."""
            )

        def _custom_nemo_callable(*args, **kwargs):
            return self._custom_nemo_callable(*args, generate_kwargs=generate_kwargs, **kwargs)

        return super().__call__(llm_api=_custom_nemo_callable, *args, **kwargs)

    @classmethod
    def from_pydantic(
        cls,
        nemorails: LLMRails,
        output_class: ModelOrListOfModels,
        *,
        prompt: Optional[str] = None,
        instructions: Optional[str] = None,
        num_reasks: Optional[int] = None,
        reask_prompt: Optional[str] = None,
        reask_instructions: Optional[str] = None,
        reask_messages: Optional[List[Dict]] = None,
        messages: Optional[List[Dict]] = None,
        tracer: Optional[Tracer] = None,
        name: Optional[str] = None,
        description: Optional[str] = None,
        output_formatter: Optional[Union[str, BaseFormatter]] = None,
    ):
        """Create a Guard instance using a Pydantic model to specify the output
        schema.

        Args:
            output_class: (Union[Type[BaseModel], List[Type[BaseModel]]]): The pydantic model that describes
            the desired structure of the output.
            prompt (str, optional): The prompt used to generate the string. Defaults to None.
            instructions (str, optional): Instructions for chat models. Defaults to None.
            reask_prompt (str, optional): An alternative prompt to use during reasks. Defaults to None.
            reask_instructions (str, optional): Alternative instructions to use during reasks. Defaults to None.
            reask_messages (List[Dict], optional): A list of messages to use during reasks. Defaults to None.
            num_reasks (int, optional): The max times to re-ask the LLM if validation fails. Deprecated
            tracer (Tracer, optional): An OpenTelemetry tracer to use for metrics and traces. Defaults to None.
            name (str, optional): A unique name for this Guard. Defaults to `gr-` + the object id.
            description (str, optional): A description for this Guard. Defaults to None.
            output_formatter (str | Formatter, optional): 'none' (default), 'jsonformer', or a Guardrails Formatter.
        """  # noqa

        if num_reasks:
            warnings.warn(
                "Setting num_reasks during initialization is deprecated"
                " and will be removed in 0.6.x!"
                "We recommend setting num_reasks when calling guard()"
                " or guard.parse() instead."
                "If you insist on setting it at the Guard level,"
                " use 'Guard.configure()'.",
                DeprecationWarning,
            )

        if reask_instructions:
            warnings.warn(
                "reask_instructions is deprecated and will be removed in 0.6.x!"
                "Please be prepared to set reask_messages instead.",
                DeprecationWarning,
            )
        if reask_prompt:
            warnings.warn(
                "reask_prompt is deprecated and will be removed in 0.6.x!"
                "Please be prepared to set reask_messages instead.",
                DeprecationWarning,
            )

        # We have to set the tracer in the ContextStore before the Rail,
        #   and therefore the Validators, are initialized
        cls._set_tracer(cls, tracer)  # type: ignore

        schema = pydantic_model_to_schema(output_class)
        exec_opts = GuardExecutionOptions(
            prompt=prompt,
            instructions=instructions,
            reask_prompt=reask_prompt,
            reask_instructions=reask_instructions,
            reask_messages=reask_messages,
            messages=messages,
        )

        # TODO: This is the only line that's changed vs the parent Guard class
        # Find a way to refactor this
        guard = cls(
            nemorails=nemorails,
            name=name,
            description=description,
            output_schema=schema.json_schema,
            validators=schema.validators,
        )
        if schema.output_type == OutputTypes.LIST:
            guard = cast(Guard[List], guard)
        else:
            guard = cast(Guard[Dict], guard)
        guard.configure(num_reasks=num_reasks, tracer=tracer)
        guard._validator_map = schema.validator_map
        guard._exec_opts = exec_opts
        guard._output_type = schema.output_type
        guard._base_model = output_class
        if isinstance(output_formatter, str):
            if isinstance(output_class, list):
                raise Exception("""Root-level arrays are not supported with the 
                jsonformer argument, but can be used with other json generation methods.
                Omit the output_formatter argument to use the other methods.""")
            output_formatter = get_formatter(
                output_formatter,
                schema=output_class.model_json_schema(),  # type: ignore
            )
        guard._output_formatter = output_formatter
        guard._fill_validators()
        return guard

    # create the callable
    def _custom_nemo_callable(self, *args, generate_kwargs, **kwargs):
        # .generate doesn't like temp
        kwargs.pop("temperature", None)

        # msg_history, messages, prompt, and instruction all may or may not be present.
        # if none of them are present, raise an error
        # if messages is present, use that
        # if msg_history is present, use 

        msg_history = kwargs.pop("msg_history", None)
        messages = kwargs.pop("messages", None)
        prompt = kwargs.pop("prompt", None)
        instructions = kwargs.pop("instructions", None)
        
        if msg_history is not None and messages is None:
            messages = msg_history

        if messages is None and msg_history is None:
            messages = []
            if instructions is not None:
                messages.append({"role": "system", "content": instructions})
            if prompt is not None:
                messages.append({"role": "system", "content": prompt})

        if messages is [] or messages is None:
            raise ValueError("messages, prompt, or instructions should be passed during a call.")
        
        # kwargs["messages"] = messages

        # return (self._nemorails.generate(**kwargs))["content"]  # type: ignore
        if not generate_kwargs:
            generate_kwargs = {}
        return (self._nemorails.generate(messages=messages, **generate_kwargs))["content"]  # type: ignore

    @deprecated(
        "This method has been deprecated. Please use the main constructor `NemoGuardrailsGuard(nemorails=nemorails)` or the `from_pydantic` method.",
    )
    def from_rail_string(cls, *args, **kwargs):
        raise NotImplementedError("""\
`from_rail_string` is not implemented for NemoguardrailsGuard.
We recommend using the main constructor `NemoGuardrailsGuard(nemorails=nemorails)`
or the `from_pydantic` method.""")

    @deprecated(
        "This method has been deprecated. Please use the main constructor `NemoGuardrailsGuard(nemorails=nemorails)` or the `from_pydantic` method.",
    )
    def from_rail(cls, *args, **kwargs):
        raise NotImplementedError("""\
`from_rail` is not implemented for NemoguardrailsGuard.
We recommend using the main constructor `NemoGuardrailsGuard(nemorails=nemorails)`
or the `from_pydantic` method.""")
