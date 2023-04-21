import logging
from string import Formatter
from typing import Callable, Dict, Optional, Tuple, Union

from eliot import start_action, to_file

from guardrails.llm_providers import PromptCallable, get_llm_ask
from guardrails.prompt import Instructions, Prompt
from guardrails.rail import Rail
from guardrails.run import Runner
from guardrails.schema import InputSchema, OutputSchema
from guardrails.utils.logs_utils import GuardState
from guardrails.utils.reask_utils import sub_reasks_with_fixed_values

logger = logging.getLogger(__name__)
to_file(open("guardrails.log", "w"))


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
    ):
        """Initialize the Guard."""
        self.rail = rail
        self.num_reasks = num_reasks
        self.guard_state = GuardState([])
        self._reask_prompt = None

    @property
    def input_schema(self) -> InputSchema:
        """Return the input schema."""
        return self.rail.input_schema

    @property
    def output_schema(self) -> OutputSchema:
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

    def __call__(
        self,
        llm_api: Callable,
        prompt_params: Dict = None,
        num_reasks: int = 1,
        *args,
        **kwargs,
    ) -> Tuple[str, Dict]:
        """Call the LLM and validate the output.

        Args:
            llm_api: The LLM API to call (e.g. openai.Completion.create)
            prompt_params: The parameters to pass to the prompt.format() method.
            num_reasks: The max times to re-ask the LLM for invalid output.
            *args: Additional arguments to pass to the LLM API.
            **kwargs: Additional keyword arguments to pass to the LLM API.

        Returns:
            The raw text output from the LLM and the validated output.
        """
        with start_action(action_type="guard_call", prompt_params=prompt_params):
            if "instructions" in kwargs:
                logger.info("Instructions overridden at call time")
                # TODO(shreya): should we overwrite self.instructions for this run?
            runner = Runner(
                instructions=kwargs.get("instructions", self.instructions),
                prompt=self.prompt,
                api=get_llm_ask(llm_api, *args, **kwargs),
                input_schema=self.input_schema,
                output_schema=self.output_schema,
                num_reasks=num_reasks,
                reask_prompt=self.reask_prompt,
            )
            guard_history = runner(prompt_params=prompt_params)
            self.guard_state = self.guard_state.push(guard_history)
            return guard_history.output, guard_history.validated_output

    def __repr__(self):
        return f"Guard(RAIL={self.rail})"

    def __rich_repr__(self):
        yield "RAIL", self.rail

    def parse(
        self,
        llm_output: str,
        llm_api: PromptCallable = None,
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
                api=get_llm_ask(llm_api, *args, **kwargs) if llm_api else None,
                input_schema=None,
                output_schema=self.output_schema,
                num_reasks=num_reasks,
                output=llm_output,
                reask_prompt=self.reask_prompt,
            )
            guard_history = runner(prompt_params=prompt_params)
            self.guard_state = self.guard_state.push(guard_history)
            return sub_reasks_with_fixed_values(guard_history.validated_output)
