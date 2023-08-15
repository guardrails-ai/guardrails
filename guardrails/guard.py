import asyncio
import logging
from string import Formatter
from typing import Any, Awaitable, Callable, Dict, List, Optional, Tuple, Union

from eliot import add_destinations, start_action
from pydantic import BaseModel

from guardrails.llm_providers import get_async_llm_ask, get_llm_ask
from guardrails.prompt import Instructions, Prompt
from guardrails.rail import Rail
from guardrails.run import AsyncRunner, Runner
from guardrails.schema import Schema
from guardrails.utils.logs_utils import GuardState
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

    def __init__(
        self,
        rail: Rail,
        num_reasks: int = 1,
        base_model: Optional[BaseModel] = None,
    ):
        """Initialize the Guard."""
        self.rail = rail
        self.num_reasks = num_reasks
        self.guard_state = GuardState([])
        self._reask_prompt = None
        self.base_model = base_model

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
        if self.prompt is None:
            return None
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
    def from_rail(cls, rail_file: str, num_reasks: int = 1) -> "Guard":
        """Create a Schema from a `.rail` file.

        Args:
            rail_file: The path to the `.rail` file.
            num_reasks: The max times to re-ask the LLM for invalid output.

        Returns:
            An instance of the `Guard` class.
        """
        return cls(Rail.from_file(rail_file), num_reasks=num_reasks)

    @classmethod
    def from_rail_string(cls, rail_string: str, num_reasks: int = 1) -> "Guard":
        """Create a Schema from a `.rail` string.

        Args:
            rail_string: The `.rail` string.
            num_reasks: The max times to re-ask the LLM for invalid output.

        Returns:
            An instance of the `Guard` class.
        """
        return cls(Rail.from_string(rail_string), num_reasks=num_reasks)

    @classmethod
    def from_pydantic(
        cls,
        output_class: BaseModel,
        prompt: Optional[str] = None,
        instructions: Optional[str] = None,
        num_reasks: int = 1,
    ) -> "Guard":
        """Create a Guard instance from a Pydantic model and prompt."""
        rail = Rail.from_pydantic(
            output_class=output_class, prompt=prompt, instructions=instructions
        )
        return cls(rail, num_reasks=num_reasks, base_model=output_class)

    def __call__(
        self,
        llm_api: Union[Callable, Callable[[Any], Awaitable[Any]]],
        prompt_params: Dict = None,
        num_reasks: Optional[int] = None,
        prompt: Optional[str] = None,
        instructions: Optional[str] = None,
        msg_history: Optional[List[Dict]] = None,
        metadata: Optional[Dict] = None,
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
        if num_reasks is None:
            num_reasks = self.num_reasks
        if metadata is None:
            metadata = {}

        # If the LLM API is async, return a coroutine
        if asyncio.iscoroutinefunction(llm_api):
            return self._call_async(
                llm_api,
                prompt_params=prompt_params,
                num_reasks=num_reasks,
                prompt=prompt,
                instructions=instructions,
                msg_history=msg_history,
                metadata=metadata,
                *args,
                **kwargs,
            )
        # Otherwise, call the LLM synchronously
        return self._call_sync(
            llm_api,
            prompt_params=prompt_params,
            num_reasks=num_reasks,
            prompt=prompt,
            instructions=instructions,
            msg_history=msg_history,
            metadata=metadata,
            *args,
            **kwargs,
        )

    def _call_sync(
        self,
        llm_api: Callable,
        prompt_params: Dict,
        num_reasks: int,
        prompt: Optional[str],
        instructions: Optional[str],
        msg_history: Optional[List[Dict]],
        metadata: Dict,
        *args,
        **kwargs,
    ) -> Tuple[str, Dict]:
        instructions = instructions or self.instructions
        prompt = prompt or self.prompt
        msg_history = msg_history or []
        if prompt is None:
            if not len(msg_history):
                raise RuntimeError(
                    "You must provide a prompt if msg_history is empty. "
                    "Alternatively, you can provide a prompt in the Schema constructor."
                )

        with start_action(action_type="guard_call", prompt_params=prompt_params):
            runner = Runner(
                instructions=instructions,
                prompt=prompt,
                msg_history=msg_history,
                api=get_llm_ask(llm_api, *args, **kwargs),
                input_schema=self.input_schema,
                output_schema=self.output_schema,
                num_reasks=num_reasks,
                metadata=metadata,
                reask_prompt=self.reask_prompt,
                base_model=self.base_model,
                guard_state=self.guard_state,
            )
            guard_history = runner(prompt_params=prompt_params)
            return guard_history.output, guard_history.validated_output

    async def _call_async(
        self,
        llm_api: Callable[[Any], Awaitable[Any]],
        prompt_params: Dict,
        num_reasks: int,
        prompt: Optional[str],
        instructions: Optional[str],
        msg_history: Optional[List[Dict]],
        metadata: Dict,
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
        instructions = instructions or self.instructions
        prompt = prompt or self.prompt
        msg_history = msg_history or []
        if prompt is None:
            if not len(msg_history):
                raise RuntimeError(
                    "You must provide a prompt if msg_history is empty. "
                    "Alternatively, you can provide a prompt in the RAIL spec."
                )
        with start_action(action_type="guard_call", prompt_params=prompt_params):
            runner = AsyncRunner(
                instructions=instructions,
                prompt=prompt,
                msg_history=msg_history,
                api=get_async_llm_ask(llm_api, *args, **kwargs),
                input_schema=self.input_schema,
                output_schema=self.output_schema,
                num_reasks=num_reasks,
                metadata=metadata,
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
        metadata: Optional[Dict] = None,
        llm_api: Union[Callable, Callable[[Any], Awaitable[Any]]] = None,
        num_reasks: int = 1,
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
        metadata = metadata or {}

        # If the LLM API is async, return a coroutine
        if asyncio.iscoroutinefunction(llm_api):
            return self._async_parse(
                llm_output,
                metadata,
                llm_api=llm_api,
                num_reasks=num_reasks,
                prompt_params=prompt_params,
                *args,
                **kwargs,
            )
        # Otherwise, call the LLM synchronously
        return self._sync_parse(
            llm_output,
            metadata,
            llm_api=llm_api,
            num_reasks=num_reasks,
            prompt_params=prompt_params,
            *args,
            **kwargs,
        )

    def _sync_parse(
        self,
        llm_output: str,
        metadata: Dict,
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
                msg_history=None,
                api=get_llm_ask(llm_api, *args, **kwargs) if llm_api else None,
                input_schema=None,
                output_schema=self.output_schema,
                num_reasks=num_reasks,
                metadata=metadata,
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
        metadata: Dict,
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
                msg_history=None,
                api=get_async_llm_ask(llm_api, *args, **kwargs) if llm_api else None,
                input_schema=None,
                output_schema=self.output_schema,
                num_reasks=num_reasks,
                metadata=metadata,
                output=llm_output,
                reask_prompt=self.reask_prompt,
                base_model=self.base_model,
                guard_state=self.guard_state,
            )
            guard_history = await runner.async_run(prompt_params=prompt_params)
            return sub_reasks_with_fixed_values(guard_history.validated_output)
