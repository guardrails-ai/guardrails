from typing import Dict, List, Union

from guardrails.prompt.prompt import Prompt

"""List[Dict[str, Union[Prompt, str]]]"""
MessageHistory = List[Dict[str, Union[Prompt, str]]]
