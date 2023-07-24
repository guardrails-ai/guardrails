import asyncio
import logging
import os
import random
import string
from string import Formatter
from typing import Any, Awaitable, Callable, Dict, List, Optional, Tuple, Union

from eliot import add_destinations, start_action
from guard_rails_api_client.models import Guard as GuardModel
from guard_rails_api_client.models import (
    History,
    HistoryEvent,
    ValidatePayload,
    ValidationOutput,
)
from pydantic import BaseModel

from guardrails.api import GuardrailsApiClient
from guardrails.llm_providers import (
    get_async_llm_ask,
    get_llm_api_enum,
    get_llm_ask,
    llm_api_is_manifest,
)
from guardrails.prompt import Instructions, Prompt
from guardrails.rail import Rail
from guardrails.run import AsyncRunner, Runner
from guardrails.schema import Schema
from guardrails.utils.logs_utils import GuardHistory, GuardLogs, GuardState, ReAsk
from guardrails.utils.reask_utils import sub_reasks_with_fixed_values

logger = logging.getLogger(__name__)
actions_logger = logging.getLogger(f"{__name__}.actions")
add_destinations(actions_logger.debug)


class Guard:
    """The Guard class.

    This class is the main entry point for using Guardrails. It is
    initialized from either `from_rail` or `from_rail_string` methods,
    which take in a `.rail` file or string, respectively. The `__call__`
    method functions as a wrapper around LLM APIs. It takes in an LLM
    API, and optional prompt parameters, and returns the raw output from
    the LLM and the validated output.
    """

    _api_client: GuardrailsApiClient = None

    def __init__(
        self,
        rail: Rail,  # TODO: Make optional next major version, allow retrieval by name
        num_reasks: int = 1,
        base_model: Optional[BaseModel] = None,
        name: Optional[str] = None,  # TODO: Make name mandatory on next major version
        openai_api_key: Optional[str] = None,
    ):
        """Initialize the Guard."""
        self.rail = rail
        self.num_reasks = num_reasks
        self.guard_state = GuardState([])
        self._reask_prompt = None
        self.base_model = base_model
        self.name = name
        self.openai_api_key = (
            openai_api_key
            if openai_api_key is not None
            else os.environ.get("OPENAI_API_KEY")
        )

        
        api_key = os.environ.get("GUARDRAILS_API_KEY")
        if api_key is not None:
            if name is None:
                self.name = "".join(
                    random.choices(string.ascii_uppercase + string.digits, k=12)
                )
                print("Warning: No name passed to guard!")
                print(
                    "Use this auto-generated name to re-use this guard: {name}".format(
                        name=self.name
                    )
                )
            self._api_client = GuardrailsApiClient(api_key=api_key)
            self.upsert_guard()

    @property
    def input_schema(self) -> Schema:
        """Return the input schema."""
        return self.rail.input_schema

    @property
    def output_schema(self) -> Schema:
        """Return the output schema."""
        return self.rail.output_schema

    @property
    def instructions(self) -> Instructions:
        """Return the instruction-prompt."""
        return self.rail.instructions

    @property
    def prompt(self) -> Prompt:
        """Return the prompt."""
        return self.rail.prompt

    @property
    def raw_prompt(self) -> Prompt:
        """Return the prompt, alias for `prompt`."""
        return self.prompt

    @property
    def base_prompt(self) -> str:
        """Return the base prompt i.e. prompt.source."""
        return self.prompt.source

    @property
    def script(self) -> Optional[Dict]:
        """Return the script."""
        return self.rail.script

    @property
    def state(self) -> GuardState:
        """Return the state."""
        return self.guard_state

    @property
    def reask_prompt(self) -> Prompt:
        """Return the reask prompt."""
        return self._reask_prompt

    @reask_prompt.setter
    def reask_prompt(self, reask_prompt: Union[str, Prompt]):
        """Set the reask prompt."""

        if isinstance(reask_prompt, str):
            reask_prompt = Prompt(reask_prompt)

        # Check that the reask prompt has the correct variables
        variables = [
            t[1] for t in Formatter().parse(reask_prompt.source) if t[1] is not None
        ]
        assert set(variables) == {"previous_response", "output_schema"}
        self._reask_prompt = reask_prompt

    def configure(
        self,
        num_reasks: int = 1,
    ):
        """Configure the Guard."""
        self.num_reasks = num_reasks

    @classmethod
    def from_rail(
        cls, rail_file: str, num_reasks: int = 1, name: Optional[str] = None
    ) -> "Guard":
        """Create a Schema from a `.rail` file.

        Args:
            rail_file: The path to the `.rail` file.
            num_reasks: The max times to re-ask the LLM for invalid output.

        Returns:
            An instance of the `Guard` class.
        """

        return cls(Rail.from_file(rail_file), num_reasks=num_reasks, name=name)

    @classmethod
    def from_rail_string(
        cls, rail_string: str, num_reasks: int = 1, name: Optional[str] = None
    ) -> "Guard":
        """Create a Schema from a `.rail` string.

        Args:
            rail_string: The `.rail` string.
            num_reasks: The max times to re-ask the LLM for invalid output.

        Returns:
            An instance of the `Guard` class.
        """
        return cls(Rail.from_string(rail_string), num_reasks=num_reasks, name=name)

    @classmethod
    def from_pydantic(
        cls,
        output_class: BaseModel,
        prompt: str,
        instructions: Optional[str] = None,
        num_reasks: int = 1,
        name: Optional[str] = None,
    ) -> "Guard":
        """Create a Guard instance from a Pydantic model and prompt."""
        rail = Rail.from_pydantic(
            output_class=output_class, prompt=prompt, instructions=instructions
        )
        return cls(rail, num_reasks=num_reasks, base_model=output_class, name=name)

    def __call__(
        self,
        llm_api: Union[Callable, Callable[[Any], Awaitable[Any]]],
        prompt_params: Dict = None,
        num_reasks: Optional[int] = None,
        *args,
        **kwargs,
    ) -> Union[Tuple[str, Dict], Awaitable[Tuple[str, Dict]]]:
        """Call the LLM and validate the output. Pass an async LLM API to
        return a coroutine.

        Args:
            llm_api: The LLM API to call
                     (e.g. openai.Completion.create or openai.Completion.acreate)
            prompt_params: The parameters to pass to the prompt.format() method.
            num_reasks: The max times to re-ask the LLM for invalid output.

        Returns:
            The raw text output from the LLM and the validated output.
        """
        if self._api_client is not None and llm_api_is_manifest(llm_api) is not True:
            # TODO: Run locally if llm_api is Manifest
            return self.validate(
                llm_api=llm_api,
                num_reasks=num_reasks,
                prompt_params=prompt_params,
                *args,
                **kwargs,
            )

        if num_reasks is None:
            num_reasks = self.num_reasks

        # If the LLM API is async, return a coroutine
        if asyncio.iscoroutinefunction(llm_api):
            return self._call_async(
                llm_api,
                prompt_params=prompt_params,
                num_reasks=num_reasks,
                *args,
                **kwargs,
            )
        # Otherwise, call the LLM synchronously
        return self._call_sync(
            llm_api,
            prompt_params=prompt_params,
            num_reasks=num_reasks,
            *args,
            **kwargs,
        )

    def _call_sync(
        self,
        llm_api: Callable,
        prompt_params: Dict = None,
        num_reasks: int = 1,
        *args,
        **kwargs,
    ) -> Tuple[str, Dict]:
        with start_action(action_type="guard_call", prompt_params=prompt_params):
            if "instructions" in kwargs:
                logger.info("Instructions overridden at call time")
                # TODO(shreya): should we overwrite self.instructions for this run?
            runner = Runner(
                instructions=kwargs.get("instructions", self.instructions),
                prompt=self.prompt,
                api=get_llm_ask(
                    llm_api, openai_api_key=self.openai_api_key, *args, **kwargs
                ),
                input_schema=self.input_schema,
                output_schema=self.output_schema,
                num_reasks=num_reasks,
                reask_prompt=self.reask_prompt,
                base_model=self.base_model,
                guard_state=self.guard_state,
            )
            guard_history = runner(prompt_params=prompt_params)
            return guard_history.output, guard_history.validated_output

    async def _call_async(
        self,
        llm_api: Callable[[Any], Awaitable[Any]],
        prompt_params: Dict = None,
        num_reasks: int = 1,
        *args,
        **kwargs,
    ) -> Tuple[str, Dict]:
        """Call the LLM asynchronously and validate the output.

        Args:
            llm_api: The LLM API to call asynchronously (e.g. openai.Completion.acreate)
            prompt_params: The parameters to pass to the prompt.format() method.
            num_reasks: The max times to re-ask the LLM for invalid output.

        Returns:
            The raw text output from the LLM and the validated output.
        """
        with start_action(action_type="guard_call", prompt_params=prompt_params):
            if "instructions" in kwargs:
                logger.info("Instructions overridden at call time")
                # TODO(shreya): should we overwrite self.instructions for this run?
            runner = AsyncRunner(
                instructions=kwargs.get("instructions", self.instructions),
                prompt=self.prompt,
                api=get_async_llm_ask(
                    llm_api, openai_api_key=self.openai_api_key * args, **kwargs
                ),
                input_schema=self.input_schema,
                output_schema=self.output_schema,
                num_reasks=num_reasks,
                reask_prompt=self.reask_prompt,
                base_model=self.base_model,
                guard_state=self.guard_state,
            )
            guard_history = await runner.async_run(prompt_params=prompt_params)
            return guard_history.output, guard_history.validated_output

    def __repr__(self):
        return f"Guard(RAIL={self.rail})"

    def __rich_repr__(self):
        yield "RAIL", self.rail

    def parse(
        self,
        llm_output: str,
        llm_api: Union[Callable, Callable[[Any], Awaitable[Any]]] = None,
        num_reasks: int = None,
        prompt_params: Dict = None,
        *args,
        **kwargs,
    ) -> Union[Tuple[str, Dict], Awaitable[Tuple[str, Dict]]]:
        """Alternate flow to using Guard where the llm_output is known.

        Args:
            llm_api: The LLM API to call
                     (e.g. openai.Completion.create or openai.Completion.acreate)
            num_reasks: The max times to re-ask the LLM for invalid output.

        Returns:
            The validated response.
        """

        if self._api_client is not None and llm_api_is_manifest(llm_api) is not True:
            return self.validate(
                llm_output=llm_output,
                llm_api=llm_api,
                num_reasks=num_reasks,
                prompt_params=prompt_params,
                *args,
                **kwargs,
            )

        if num_reasks is None:
            num_reasks = self.num_reasks if self.num_reasks is not None else 1

        # If the LLM API is async, return a coroutine
        if asyncio.iscoroutinefunction(llm_api):
            return self._async_parse(
                llm_output,
                llm_api=llm_api,
                num_reasks=num_reasks,
                prompt_params=prompt_params,
                *args,
                **kwargs,
            )
        # Otherwise, call the LLM synchronously
        return self._sync_parse(
            llm_output,
            llm_api=llm_api,
            num_reasks=num_reasks,
            prompt_params=prompt_params,
            *args,
            **kwargs,
        )

    def _sync_parse(
        self,
        llm_output: str,
        llm_api: Callable = None,
        num_reasks: int = 1,
        prompt_params: Dict = None,
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
        with start_action(action_type="guard_parse"):
            runner = Runner(
                instructions=None,
                prompt=None,
                api=get_llm_ask(
                    llm_api, openai_api_key=self.openai_api_key, *args, **kwargs
                )
                if llm_api
                else None,
                input_schema=None,
                output_schema=self.output_schema,
                num_reasks=num_reasks,
                output=llm_output,
                reask_prompt=self.reask_prompt,
                base_model=self.base_model,
                guard_state=self.guard_state,
            )
            guard_history = runner(prompt_params=prompt_params)
            return sub_reasks_with_fixed_values(guard_history.validated_output)

    async def _async_parse(
        self,
        llm_output: str,
        llm_api: Callable[[Any], Awaitable[Any]] = None,
        num_reasks: int = 1,
        prompt_params: Dict = None,
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
        with start_action(action_type="guard_parse"):
            runner = AsyncRunner(
                instructions=None,
                prompt=None,
                api=get_async_llm_ask(
                    llm_api, openai_api_key=self.openai_api_key, *args, **kwargs
                )
                if llm_api
                else None,
                input_schema=None,
                output_schema=self.output_schema,
                num_reasks=num_reasks,
                output=llm_output,
                reask_prompt=self.reask_prompt,
                base_model=self.base_model,
                guard_state=self.guard_state,
            )
            guard_history = await runner.async_run(prompt_params=prompt_params)
            return sub_reasks_with_fixed_values(guard_history.validated_output)

    def _to_request(self) -> Dict:
        return {
            "name": self.name,
            "railspec": self.rail._to_request(),
            "numReasks": self.num_reasks,
        }

    def upsert_guard(self):
        guard_dict = self._to_request()
        self._api_client.upsert_guard(GuardModel.from_dict(guard_dict))

    def validate(
        self,
        llm_output: Optional[str] = None,
        llm_api: Union[Callable, Callable[[Any], Awaitable[Any]]] = None,
        num_reasks: int = None,
        prompt_params: Dict = None,
        *args,
        **kwargs,
    ):
        payload = {"args": list(args)}
        payload.update(**kwargs)
        if llm_output is not None:
            payload["llmOutput"] = llm_output
        if num_reasks is not None:
            payload["numReasks"] = num_reasks
        if prompt_params is not None:
            payload["promptParams"] = prompt_params
        if llm_api is not None:
            payload["llmApi"] = get_llm_api_enum(llm_api)
        # TODO: get enum for llm_api
        validation_output: ValidationOutput = self._api_client.validate(
            guard=self,
            payload=ValidatePayload.from_dict(payload),
            openai_api_key=self.openai_api_key,
        )

        session_history = (
            validation_output.session_history
            if validation_output is not None and validation_output.session_history
            else []
        )
        history: History
        for history in session_history:
            history_events: List[HistoryEvent] = history.history
            if history_events is None:
                continue

            history_logs = [
                GuardLogs(
                    instructions=h.instructions,
                    output=h.output,
                    parsed_output=h.parsed_output.to_dict(),
                    prompt=Prompt(h.prompt.source)
                    if h.prompt.source is not None
                    else None,
                    reasks=[
                        ReAsk(
                            incorrect_value=r.to_dict().get("incorrect_value"),
                            error_message=r.to_dict().get("error_message"),
                            fix_value=r.to_dict().get("fix_value"),
                            path=r.to_dict().get("path"),
                        )
                        for r in h.reasks
                    ],
                    validated_output=h.validated_output.to_dict(),
                )
                for h in history_events
            ]
            self.guard_state = self.guard_state.push(GuardHistory(history=history_logs))

        if llm_output is not None:
            return validation_output.validated_output
        else:
            return (
                validation_output.raw_llm_response,
                validation_output.validated_output,
            )
