from typing import Callable, Dict, Optional, Tuple, Any
from guardrails.llm_providers import PromptCallable
from guardrails.schema import Schema
from guardrails.prompt import Instructions, Prompt

class Callback():

    def before_prepare(
        self,
        index: int,
        instructions: Optional[Instructions],
        prompt: Prompt,
        prompt_params: Dict,
        api: PromptCallable,
        input_schema: Schema,
        output_schema: Schema,
    ) -> Tuple[Instructions, Prompt]:
        ...

    def after_prepare(
        self,
        instructions: Optional[Instructions],
        prompt: Prompt,
    ) -> Any:
        ...

    def before_call(
        self,
        index: int,
        instructions: Optional[Instructions],
        prompt: Prompt,
        api: Callable,
        output: str = None,
    ) -> str:
        ...

    def after_call(
        self,
        input: str
    ) -> Any:
        ...
    
    def before_parse(
        self,
        index: int,
        output: str,
        output_schema: Schema,
    ) -> Any:
        ...


    def after_parse(
        self,
        *args: Any,
        **kwargs: Any
    ) -> Any:
        ...


    def before_validate(
        self,
        index: int,
        parsed_output: Any,
        output_schema: Schema,
    ) -> Any:
        ...


    def after_validate(
        self,
        *args: Any,
        **kwargs: Any
    ) -> Any:
        ...