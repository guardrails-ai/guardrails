from typing import Callable, Iterable, Optional, Union
from typing_extensions import deprecated

from guardrails.classes.output_type import OT
from guardrails.classes.validation_outcome import ValidationOutcome

from guardrails import Guard
from nemoguardrails import LLMRails


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
        self, llm_api: Optional[Callable] = None, *args, **kwargs
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

        # create the callable
        def custom_callable(**kwargs):
            # .generate doesn't like temp
            kwargs.pop("temperature", None)

            msg_history = kwargs.pop("msg_history", None)
            messages = (
                msg_history
                if kwargs.get("messages") is None
                else kwargs.get("messages")
            )

            prompt = kwargs.get("prompt")

            if messages is not None:
                kwargs["messages"] = messages

            if (messages is None) and (prompt is None):
                raise ValueError("""messages or prompt should be passed.""")

            return (self._nemorails.generate(**kwargs))["content"]  # type: ignore

        return super().__call__(llm_api=custom_callable, *args, **kwargs)

    def from_pydantic(self, *args, **kwargs):
        pass

    @deprecated(
        "This method has been deprecated. Please use the main constructor `NemoGuardrailsGuard(nemorails=nemorails)` or the `from_pydantic` method.",
    )
    def from_rail_string(cls, *args, **kwargs):
        raise NotImplementedError("""\
`from_rail_string` is not implemented for NemoguardrailsGuard.
We recommend using the main constructor `NemoGuardrailsGuard(nemorails=nemorails)`
or the `from_pydantic` method.""")
