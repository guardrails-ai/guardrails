from llama_index.query_pipeline import (
    CustomQueryComponent,
    InputKeys,
    OutputKeys,
)
from typing import Dict, Any
from llama_index.llms.llm import BaseLLM
from pydantic import Field
from guardrails.functional.guard import Guard as FGuard

class Guard(FGuard, CustomQueryComponent):
    pass