from typing import Dict, List, Union

from guardrails.prompt.prompt import Prompt


Messages = List[Dict[str, Union[Prompt, str]]]
