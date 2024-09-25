from typing import Dict, List, Optional


def mock_llm(
    messages,
    *args,
    **kwargs,
) -> str:
    return ""


async def mock_async_llm(
    messages,
    *args,
    **kwargs,
) -> str:
    return ""
