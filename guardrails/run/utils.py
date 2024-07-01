import copy
from typing import Dict, cast

from guardrails.prompt.prompt import Prompt
from guardrails.prompt.messages import Messages
from guardrails.types.inputs import MessageHistory


def msg_history_source(msg_history: MessageHistory) -> MessageHistory:
    msg_history_copy = []
    for msg in msg_history:
        msg_copy = copy.deepcopy(msg)
        content = (
            msg["content"].source
            if isinstance(msg["content"], Prompt)
            else msg["content"]
        )
        msg_copy["content"] = content
        msg_history_copy.append(cast(Dict[str, str], msg_copy))
    return msg_history_copy


def msg_history_string(msg_history: MessageHistory) -> str:
    msg_history_copy = ""
    for msg in msg_history:
        content = (
            msg["content"].source
            if isinstance(msg["content"], Prompt)
            else msg["content"]
        )
        msg_history_copy += content
    return msg_history_copy


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
