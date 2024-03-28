import copy
from typing import Dict, List


def msg_history_source(msg_history) -> List[Dict[str, str]]:
    msg_history_copy = copy.deepcopy(msg_history)
    for msg in msg_history_copy:
        msg["content"] = msg["content"].source
    return msg_history_copy


def msg_history_string(msg_history) -> str:
    msg_history_copy = ""
    for msg in msg_history:
        msg_history_copy += msg["content"].source
    return msg_history_copy
