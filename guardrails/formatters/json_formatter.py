import json

from jsonformer import Jsonformer

from guardrails.formatters.base_formatter import BaseFormatter
from guardrails.llm_providers import (
    ArbitraryCallable,
    HuggingFacePipelineCallable,
    HuggingFaceModelCallable,
)


class JsonFormatter(BaseFormatter):
    def __init__(self, schema: dict):
        self.output_schema = schema

    def wrap_callable(self, llm_callable) -> ArbitraryCallable:
        # JSON Schema enforcement experiment.
        if isinstance(llm_callable, HuggingFacePipelineCallable):
            model = llm_callable.init_kwargs["pipeline"]
            return ArbitraryCallable(
                lambda p: json.dumps(
                    Jsonformer(
                        model=model.model,
                        tokenizer=model.tokenizer,
                        json_schema=self.output_schema,
                        prompt=p,
                    )()
                )
            )
        elif isinstance(llm_callable, HuggingFaceModelCallable):
            # This will not work because 'model_generate' is the .gen method.
            # model = self.api.init_kwargs["model_generate"]
            # Use the __self__ to grab the base mode for passing into JF.
            model = llm_callable.init_kwargs["model_generate"].__self__
            tokenizer = llm_callable.init_kwargs["tokenizer"]
            return ArbitraryCallable(
                lambda p: json.dumps(
                    Jsonformer(
                        model=model,
                        tokenizer=tokenizer,
                        json_schema=self.output_schema,
                        prompt=p,
                    )()
                )
            )
        else:
            raise ValueError(
                "JsonFormatter can only be used with HuggingFace*Callable."
            )
