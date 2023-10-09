import asyncio
import contextvars
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
from guardrails.validators import Validator

logger = logging.getLogger(__name__)
actions_logger = logging.getLogger(f"{__name__}.actions")
add_destinations(actions_logger.debug)


class Guard:
    """The Guard class.

    This class is the main entry point for using Guardrails. It is
    initialized from one of the following class methods:

    - `from_rail`
    - `from_rail_string`
    - `from_pydantic`
    - `from_string`

    The `__call__`
    method functions as a wrapper around LLM APIs. It takes in an LLM
    API, and optional prompt parameters, and returns the raw output from
    the LLM and the validated output.
    """

    def __init__(
        self,
        rail: Rail,
        num_reasks: int = None,
        base_model: Optional[BaseModel] = None,
    ):
        """Initialize the Guard."""
        self.rail = rail
        self.num_reasks = num_reasks
        self.guard_state = GuardState(all_histories=[])
        self._reask_prompt = None
        self._reask_instructions = None
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
        variable_set = set(variables)
        assert variable_set.__contains__("previous_response")
        assert variable_set.__contains__("output_schema")
        self._reask_prompt = reask_prompt

    @property
    def reask_instructions(self) -> Prompt:
        """Return the reask prompt."""
        return self._reask_instructions

    @reask_instructions.setter
    def reask_instructions(self, reask_instructions: Union[str, Instructions]):
        """Set the reask prompt."""

        if isinstance(reask_instructions, str):
            reask_instructions = Instructions(reask_instructions)

        self._reask_instructions = reask_instructions

    def configure(
        self,
        num_reasks: int = None,
    ):
        """Configure the Guard."""
        self.num_reasks = (
            num_reasks
            if num_reasks is not None
            else self.num_reasks
            if self.num_reasks is not None
            else 1
        )

    @classmethod
    def from_rail(cls, rail_file: str, num_reasks: int = None) -> "Guard":
        """Create a Schema from a `.rail` file.

        Args:
            rail_file: The path to the `.rail` file.
            num_reasks: The max times to re-ask the LLM for invalid output.

        Returns:
            An instance of the `Guard` class.
        """
        return cls(Rail.from_file(rail_file), num_reasks=num_reasks)

    @classmethod
    def from_rail_string(cls, rail_string: str, num_reasks: int = None) -> "Guard":
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
        num_reasks: int = None,
    ) -> "Guard":
        """Create a Guard instance from a Pydantic model and prompt."""
        rail = Rail.from_pydantic(
            output_class=output_class, prompt=prompt, instructions=instructions
        )
        return cls(rail, num_reasks=num_reasks, base_model=output_class)

    @classmethod
    def from_string(
        cls,
        validators: List[Validator],
        description: Optional[str] = None,
        prompt: Optional[str] = None,
        instructions: Optional[str] = None,
        reask_prompt: Optional[str] = None,
        reask_instructions: Optional[str] = None,
        num_reasks: int = None,
    ) -> "Guard":
        """Create a Guard instance for a string response with prompt,
        instructions, and validations.

        Parameters: Arguments
            validators: (List[Validator]): The list of validators to apply to the string output.
            description (str, optional): A description for the string to be generated. Defaults to None.
            prompt (str, optional): The prompt used to generate the string. Defaults to None.
            instructions (str, optional): Instructions for chat models. Defaults to None.
            reask_prompt (str, optional): An alternative prompt to use during reasks. Defaults to None.
            reask_instructions (str, optional): Alternative instructions to use during reasks. Defaults to None.
            num_reasks (int, optional): The max times to re-ask the LLM for invalid output.
        """  # noqa
        rail = Rail.from_string_validators(
            validators=validators,
            description=description,
            prompt=prompt,
            instructions=instructions,
            reask_prompt=reask_prompt,
            reask_instructions=reask_instructions,
        )
        return cls(rail, num_reasks=num_reasks)

    def __call__(
        self,
        llm_api: Union[Callable, Callable[[Any], Awaitable[Any]]],
        prompt_params: Dict = None,
        num_reasks: Optional[int] = None,
        prompt: Optional[str] = None,
        instructions: Optional[str] = None,
        msg_history: Optional[List[Dict]] = None,
        metadata: Optional[Dict] = None,
        full_schema_reask: Optional[bool] = None,
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
            prompt: The prompt to use for the LLM.
            instructions: Instructions for chat models.
            msg_history: The message history to pass to the LLM.
            metadata: Metadata to pass to the validators.
            full_schema_reask: When reasking, whether to regenerate the full schema
                               or just the incorrect values.
                               Defaults to `True` if a base model is provided,
                               `False` otherwise.

        Returns:
            The raw text output from the LLM and the validated output.
        """
        self.configure(num_reasks)
        if metadata is None:
            metadata = {}
        if full_schema_reask is None:
            full_schema_reask = self.base_model is not None

        context = contextvars.ContextVar("kwargs")
        context.set(kwargs)

        # If the LLM API is async, return a coroutine
        if asyncio.iscoroutinefunction(llm_api):
            return self._call_async(
                llm_api,
                prompt_params=prompt_params,
                num_reasks=self.num_reasks,
                prompt=prompt,
                instructions=instructions,
                msg_history=msg_history,
                metadata=metadata,
                full_schema_reask=full_schema_reask,
                *args,
                **kwargs,
            )
        # Otherwise, call the LLM synchronously
        return self._call_sync(
            llm_api,
            prompt_params=prompt_params,
            num_reasks=self.num_reasks,
            prompt=prompt,
            instructions=instructions,
            msg_history=msg_history,
            metadata=metadata,
            full_schema_reask=full_schema_reask,
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
        full_schema_reask: bool,
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
                reask_instructions=self.reask_instructions,
                base_model=self.base_model,
                guard_state=self.guard_state,
                full_schema_reask=full_schema_reask,
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
        full_schema_reask: bool,
        *args,
        **kwargs,
    ) -> Tuple[str, Dict]:
        """Call the LLM asynchronously and validate the output.

        Args:
            llm_api: The LLM API to call asynchronously (e.g. openai.Completion.acreate)
            prompt_params: The parameters to pass to the prompt.format() method.
            num_reasks: The max times to re-ask the LLM for invalid output.
            prompt: The prompt to use for the LLM.
            instructions: Instructions for chat models.
            msg_history: The message history to pass to the LLM.
            metadata: Metadata to pass to the validators.
            full_schema_reask: When reasking, whether to regenerate the full schema
                               or just the incorrect values.
                               Defaults to `True` if a base model is provided,
                               `False` otherwise.

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
                reask_instructions=self.reask_instructions,
                base_model=self.base_model,
                guard_state=self.guard_state,
                full_schema_reask=full_schema_reask,
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
        num_reasks: int = None,
        prompt_params: Dict = None,
        full_schema_reask: bool = None,
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
        num_reasks = (
            num_reasks if num_reasks is not None else 0 if llm_api is None else None
        )
        self.configure(num_reasks)
        if full_schema_reask is None:
            full_schema_reask = self.base_model is not None
        metadata = metadata or {}

        context = contextvars.ContextVar("kwargs")
        context.set(kwargs)

        # If the LLM API is async, return a coroutine
        if asyncio.iscoroutinefunction(llm_api):
            return self._async_parse(
                llm_output,
                metadata,
                llm_api=llm_api,
                num_reasks=self.num_reasks,
                prompt_params=prompt_params,
                full_schema_reask=full_schema_reask,
                *args,
                **kwargs,
            )
        # Otherwise, call the LLM synchronously
        return self._sync_parse(
            llm_output,
            metadata,
            llm_api=llm_api,
            num_reasks=self.num_reasks,
            prompt_params=prompt_params,
            full_schema_reask=full_schema_reask,
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
        full_schema_reask: bool = False,
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
                instructions=kwargs.pop("instructions", None),
                prompt=kwargs.pop("prompt", None),
                msg_history=kwargs.pop("msg_history", None),
                api=get_llm_ask(llm_api, *args, **kwargs) if llm_api else None,
                input_schema=None,
                output_schema=self.output_schema,
                num_reasks=num_reasks,
                metadata=metadata,
                output=llm_output,
                reask_prompt=self.reask_prompt,
                reask_instructions=self.reask_instructions,
                base_model=self.base_model,
                guard_state=self.guard_state,
                full_schema_reask=full_schema_reask,
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
        full_schema_reask: bool = False,
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
                instructions=kwargs.pop("instructions", None),
                prompt=kwargs.pop("prompt", None),
                msg_history=kwargs.pop("msg_history", None),
                api=get_async_llm_ask(llm_api, *args, **kwargs) if llm_api else None,
                input_schema=None,
                output_schema=self.output_schema,
                num_reasks=num_reasks,
                metadata=metadata,
                output=llm_output,
                reask_prompt=self.reask_prompt,
                reask_instructions=self.reask_instructions,
                base_model=self.base_model,
                guard_state=self.guard_state,
                full_schema_reask=full_schema_reask,
            )
            guard_history = await runner.async_run(prompt_params=prompt_params)
            return sub_reasks_with_fixed_values(guard_history.validated_output)
