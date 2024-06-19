from typing import Dict, List, Union

from guardrails.prompt.prompt import Prompt


MessageHistory = List[Dict[str, Union[Prompt, str]]]
