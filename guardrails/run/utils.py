import copy
from typing import Dict, List

from guardrails.types.inputs import MessageHistory
# todo move this to types
from guardrails.messages.messages import Messages


def messages_source(messages: Messages) -> List[Dict[str, str]]:
    messages_copy = copy.deepcopy(messages)
    for msg in messages_copy:
        msg["content"] = msg["content"]
    return messages_copy

def messages_string(messages: Messages) -> str:
    messages_copy = ""
    for msg in messages:
        messages_copy += msg["content"]
    return messages_copy

def msg_history_source(msg_history: MessageHistory) -> List[Dict[str, str]]:
    msg_history_copy = copy.deepcopy(msg_history)
    for msg in msg_history_copy:
        msg["content"] = msg["content"].source
    return msg_history_copy


def msg_history_string(msg_history: MessageHistory) -> str:
    msg_history_copy = ""
    for msg in msg_history:
        msg_history_copy += msg["content"].source
    return msg_history_copy
