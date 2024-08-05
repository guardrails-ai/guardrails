from typing import Dict, List, Optional
from dataclasses import dataclass


@dataclass
class GuardExecutionOptions:
    prompt: Optional[str] = None
    instructions: Optional[str] = None
    msg_history: Optional[List[Dict]] = None
    messages: Optional[List[Dict]] = None
    reask_prompt: Optional[str] = None
    reask_instructions: Optional[str] = None
    reask_messages: Optional[List[Dict]] = None
    num_reasks: Optional[int] = None
