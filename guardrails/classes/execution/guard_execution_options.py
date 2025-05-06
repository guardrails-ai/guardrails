from typing import Dict, List, Optional
from dataclasses import dataclass


@dataclass
class GuardExecutionOptions:
    messages: Optional[List[Dict]] = None
    reask_messages: Optional[List[Dict]] = None
    num_reasks: Optional[int] = None
