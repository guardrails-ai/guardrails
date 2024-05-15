from typing import Optional
from dataclasses import dataclass


@dataclass
class GuardExecutionOptions:
    prompt: Optional[str] = None
    instructions: Optional[str] = None
    reask_prompt: Optional[str] = None
    reask_instructions: Optional[str] = None
    num_reasks: Optional[int] = None
    prompt_schema: Optional[str] = None
    instructions_schema: Optional[str] = None
    msg_history_schema: Optional[str] = None
