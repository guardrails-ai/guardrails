from typing import Dict, List, Optional


def mock_llm(
    prompt: Optional[str] = None,
    *args,
    instructions: Optional[str] = None,
    msg_history: Optional[List[Dict[str, str]]] = None,
    **kwargs,
) -> str:
    return ""


async def mock_async_llm(
    prompt: Optional[str] = None,
    *args,
    instructions: Optional[str] = None,
    msg_history: Optional[List[Dict[str, str]]] = None,
    **kwargs,
) -> str:
    return ""
