from typing import Dict, List, Optional


def mock_llm(
    *args,
    messages: Optional[List[Dict[str, str]]] = None,
    **kwargs,
) -> str:
    return ""


async def mock_async_llm(
    *args,
    messages: Optional[List[Dict[str, str]]] = None,
    **kwargs,
) -> str:
    return ""
