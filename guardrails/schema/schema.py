import pprint
from typing import TYPE_CHECKING, Any, Dict, Optional, Sequence, Set, Tuple, Union

from lxml import etree as ET
from typing_extensions import Self

from guardrails.classes.history import Iteration
from guardrails.datatypes import DataType
from guardrails.llm_providers import PromptCallableBase
from guardrails.prompt import Instructions, Prompt
from guardrails.utils.reask_utils import ReAsk

if TYPE_CHECKING:
    pass


class Schema:
    """Schema class that holds a _schema attribute."""

    reask_prompt_vars: Set[str]

    def __init__(
        self,
        schema: DataType,
        reask_prompt_template: Optional[str] = None,
        reask_instructions_template: Optional[str] = None,
    ) -> None:
        self.root_datatype = schema

        # Setup reask templates
        self.reask_prompt_template = reask_prompt_template
        self.reask_instructions_template = reask_instructions_template

    @classmethod
    def from_xml(
        cls,
        root: ET._Element,
        reask_prompt_template: Optional[str] = None,
        reask_instructions_template: Optional[str] = None,
    ) -> Self:
        """Create a schema from an XML element."""
        raise NotImplementedError

    def __repr__(self) -> str:
        # FIXME make sure this is pretty
        return f"{self.__class__.__name__}({pprint.pformat(self.root_datatype)})"

    @property
    def reask_prompt_template(self) -> Optional[Prompt]:
        return self._reask_prompt_template

    @reask_prompt_template.setter
    def reask_prompt_template(self, value: Optional[str]) -> None:
        self.check_valid_reask_prompt(value)
        if value is not None:
            self._reask_prompt_template = Prompt(value)
        else:
            self._reask_prompt_template = None

    @property
    def reask_instructions_template(self) -> Optional[Instructions]:
        return self._reask_instructions_template

    @reask_instructions_template.setter
    def reask_instructions_template(self, value: Optional[str]) -> None:
        if value is not None:
            self._reask_instructions_template = Instructions(value)
        else:
            self._reask_instructions_template = None

    def validate(
        self,
        iteration: Iteration,
        data: Any,
        metadata: Dict,
        attempt_number: int = 0,
        **kwargs,
    ) -> Any:
        """Validate a dictionary of data against the schema.

        Args:
            data: The data to validate.

        Returns:
            The validated data.
        """
        raise NotImplementedError

    async def async_validate(
        self, iteration: Iteration, data: Any, metadata: Dict, attempt_number: int = 0
    ) -> Any:
        """Asynchronously validate a dictionary of data against the schema.

        Args:
            data: The data to validate.

        Returns:
            The validated data.
        """
        raise NotImplementedError

    def transpile(self, method: str = "default") -> str:
        """Convert the XML schema to a string that is used for prompting a
        large language model.

        Returns:
            The prompt.
        """
        raise NotImplementedError

    def parse(self, output: str, **kwargs) -> Tuple[Any, Optional[Exception]]:
        """Parse the output from the large language model.

        Args:
            output: The output from the large language model.

        Returns:
            The parsed output, and the exception that was raised (if any).
        """
        raise NotImplementedError

    def introspect(
        self, data: Any
    ) -> Tuple[Sequence[ReAsk], Optional[Union[str, Dict]]]:
        """Inspect the data for reasks.

        Args:
            data: The data to introspect.

        Returns:
            A list of ReAsk objects.
        """
        raise NotImplementedError

    def get_reask_setup(
        self,
        reasks: Sequence[ReAsk],
        original_response: Any,
        use_full_schema: bool,
        prompt_params: Optional[Dict[str, Any]] = None,
    ) -> Tuple["Schema", Prompt, Instructions]:
        """Construct a schema for reasking, and a prompt for reasking.

        Args:
            reasks: List of tuples, where each tuple contains the path to the
                reasked element, and the ReAsk object (which contains the error
                message describing why the reask is necessary).
            original_response: The value that was returned from the API, with reasks.
            use_full_schema: Whether to use the full schema, or only the schema
                for the reasked elements.

        Returns:
            The schema for reasking, and the prompt for reasking.
        """
        raise NotImplementedError

    def preprocess_prompt(
        self,
        prompt_callable: PromptCallableBase,
        instructions: Optional[Instructions],
        prompt: Prompt,
    ):
        """Preprocess the instructions and prompt before sending it to the
        model.

        Args:
            prompt_callable: The callable to be used to prompt the model.
            instructions: The instructions to preprocess.
            prompt: The prompt to preprocess.
        """
        raise NotImplementedError

    def check_valid_reask_prompt(self, reask_prompt: Optional[str]) -> None:
        if reask_prompt is None:
            return

        # Check that the reask prompt has the correct variables

        # TODO decide how to check this
        # variables = get_template_variables(reask_prompt)
        # assert set(variables) == self.reask_prompt_vars
