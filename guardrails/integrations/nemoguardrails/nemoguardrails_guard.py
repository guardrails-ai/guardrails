from typing import Any, Callable, Dict, Generic, Iterable, List, Optional, Union, cast
from typing_extensions import deprecated

from guardrails.classes.output_type import OT, OutputTypes
from guardrails.classes.validation_outcome import ValidationOutcome
from guardrails.classes.validation.validator_reference import ValidatorReference

from guardrails import Guard

from guardrails.formatters.base_formatter import BaseFormatter
from guardrails.types.pydantic import ModelOrListOfModels

from guardrails.stores.context import Tracer

try:
    from nemoguardrails import LLMRails
except ImportError:
    raise ImportError(
        "Could not import nemoguardrails, please install it with "
        "`pip install nemoguardrails`."
    )


class NemoguardrailsGuard(Guard, Generic[OT]):
    def __init__(
        self,
        nemorails: LLMRails,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self._nemorails = nemorails

    def __call__(
        self,
        llm_api: Optional[Callable] = None,
        generate_kwargs: Optional[Dict] = None,
        *args,
        **kwargs,
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
            return self._custom_nemo_callable(
                *args, generate_kwargs=generate_kwargs, **kwargs
            )

        return super().__call__(llm_api=_custom_nemo_callable, *args, **kwargs)

    @classmethod
    def _init_guard_for_cls_method(
        cls,
        *,
        name: Optional[str] = None,
        description: Optional[str] = None,
        validators: Optional[List[ValidatorReference]] = None,
        output_schema: Optional[Dict[str, Any]] = None,
        nemorails: LLMRails,
        **kwargs,
    ):
        return cls(
            nemorails,
            name=name,
            description=description,
            output_schema=output_schema,
            validators=validators,
        )

    @classmethod
    def for_pydantic(
        cls,
        output_class: ModelOrListOfModels,
        nemorails: LLMRails,
        *,
        num_reasks: Optional[int] = None,
        reask_messages: Optional[List[Dict]] = None,
        messages: Optional[List[Dict]] = None,
        tracer: Optional[Tracer] = None,
        name: Optional[str] = None,
        description: Optional[str] = None,
        output_formatter: Optional[Union[str, BaseFormatter]] = None,
        **kwargs,
    ):
        guard = super().for_pydantic(
            output_class,
            num_reasks=num_reasks,
            messages=messages,
            reask_messages=reask_messages,
            tracer=tracer,
            name=name,
            description=description,
            output_formatter=output_formatter,
            nemorails=nemorails,
        )
        if guard._output_type == OutputTypes.LIST:
            return cast(NemoguardrailsGuard[List], guard)
        else:
            return cast(NemoguardrailsGuard[Dict], guard)

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

        if messages == [] or messages is None:
            raise ValueError(
                "messages, prompt, or instructions should be passed during a call."
            )

        # kwargs["messages"] = messages

        # return (self._nemorails.generate(**kwargs))["content"]  # type: ignore
        if not generate_kwargs:
            generate_kwargs = {}
        return (self._nemorails.generate(messages=messages, **generate_kwargs))[  # type: ignore
            "content"
        ]

    @deprecated(
        "Use `for_rail_string` instead. This method will be removed in 0.6.x.",
        category=None,
    )
    @classmethod
    def from_rail_string(cls, *args, **kwargs):
        raise NotImplementedError("""\
`from_rail_string` is not implemented for NemoguardrailsGuard.
We recommend using the main constructor `NemoGuardrailsGuard(nemorails=nemorails)`
or the `from_pydantic` method.""")

    @classmethod
    def for_rail_string(cls, *args, **kwargs):
        raise NotImplementedError("""\
`for_rail_string` is not implemented for NemoguardrailsGuard.
We recommend using the main constructor `NemoGuardrailsGuard(nemorails=nemorails)`
or the `from_pydantic` method.""")

    @deprecated(
        "Use `for_rail` instead. This method will be removed in 0.6.x.",
        category=None,
    )
    @classmethod
    def from_rail(cls, *args, **kwargs):
        raise NotImplementedError("""\
`from_rail` is not implemented for NemoguardrailsGuard.
We recommend using the main constructor `NemoGuardrailsGuard(nemorails=nemorails)`
or the `from_pydantic` method.""")

    @classmethod
    def for_rail(cls, *args, **kwargs):
        raise NotImplementedError("""\
`for_rail` is not implemented for NemoguardrailsGuard.
We recommend using the main constructor `NemoGuardrailsGuard(nemorails=nemorails)`
or the `from_pydantic` method.""")
