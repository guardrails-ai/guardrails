import litellm
from typing import Any, Awaitable, Callable, Union
from guardrails_api_client.models.llm_resource import LLMResource


def get_llm_callable(
    llm_api: str,
) -> Union[Callable, Callable[[Any], Awaitable[Any]]]:
    # TODO: Add error handling and throw 400
    # do we need this anymore if were going to use the default handling
    # and only set model
    if llm_api == LLMResource.LITELLM_DOT_COMPLETION.value:
        return litellm.completion
    elif llm_api == LLMResource.LITELLM_DOT_ACOMPLETION.value:
        return litellm.acompletion
    else:
        pass
