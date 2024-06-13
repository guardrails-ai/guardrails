import copy
from typing import Dict, cast

from guardrails.prompt.prompt import Prompt
from guardrails.types.inputs import Messages

def messages_source(messages: Messages) -> Messages:
    messages_copy = []
    for msg in messages:
        msg_copy = copy.deepcopy(msg)
        content = (
            msg["content"].source
            if isinstance(msg["content"], Prompt)
            else msg["content"]
        )
        msg_copy["content"] = content
        messages_copy.append(cast(Dict[str, str], msg_copy))
    return messages_copy


def messages_string(messages: Messages) -> str:
    messages_copy = ""
    for msg in messages:
        content = (
            msg["content"].source
            if isinstance(msg["content"], Prompt)
            else msg["content"]
        )
        messages_copy += content
    return messages_copy
