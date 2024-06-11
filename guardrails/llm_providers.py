import asyncio
import warnings

from typing import (
    Any,
    Awaitable,
    Callable,
    Dict,
    Iterable,
    List,
    Optional,
    Type,
    Union,
    cast,
)

from guardrails_api_client.models import LLMResource
from pydantic import BaseModel

from guardrails.errors import UserFacingException
from guardrails.classes.llm.llm_response import LLMResponse
from guardrails.utils.openai_utils import (
    AsyncOpenAIClient,
    OpenAIClient,
    get_static_openai_acreate_func,
    get_static_openai_chat_acreate_func,
    get_static_openai_chat_create_func,
    get_static_openai_create_func,
)
from guardrails.utils.pydantic_utils import convert_pydantic_model_to_openai_fn
from guardrails.utils.safe_get import safe_get


class PromptCallableException(Exception):
    pass


###
# Synchronous wrappers
###


class PromptCallableBase:
    """A wrapper around a callable that takes in a prompt.

    Catches exceptions to let the user know clearly if the callable
    failed, and how to fix it.
    """

    supports_base_model = False

    def __init__(self, *args, **kwargs):
        self.init_args = args
        self.init_kwargs = kwargs

    def _invoke_llm(self, *args, **kwargs) -> LLMResponse:
        raise NotImplementedError

    def __call__(self, *args, **kwargs) -> LLMResponse:
        try:
            result = self._invoke_llm(
                *self.init_args, *args, **self.init_kwargs, **kwargs
            )
        except Exception as e:
            raise PromptCallableException(
                "The callable `fn` passed to `Guard(fn, ...)` failed"
                f" with the following error: `{e}`. "
                "Make sure that `fn` can be called as a function that"
                " takes in a single prompt string "
                "and returns a string."
            )
        if not isinstance(result, LLMResponse):
            raise PromptCallableException(
                "The callable `fn` passed to `Guard(fn, ...)` returned"
                f" a non-string value: {result}. "
                "Make sure that `fn` can be called as a function that"
                " takes in a single prompt string "
                "and returns a string."
            )
        return result

def chat_prompt(
    prompt: Optional[str],
    instructions: Optional[str] = None,
    msg_history: Optional[List[Dict]] = None,
) -> List[Dict[str, str]]:
    """Prepare final prompt for chat engine."""
    if msg_history:
        return msg_history
    if prompt is None:
        raise PromptCallableException(
            "You must pass in either `text` or `msg_history` to `guard.__call__`."
        )

    if not instructions:
        instructions = "You are a helpful assistant."

    return [
        {"role": "system", "content": instructions},
        {"role": "user", "content": prompt},
    ]

def nonchat_prompt(prompt: str, instructions: Optional[str] = None) -> str:
    """Prepare final prompt for nonchat engine."""
    if instructions:
        prompt = "\n\n".join([instructions, prompt])
    return prompt


class OpenAIModel(PromptCallableBase):
    pass


class ManifestCallable(PromptCallableBase):
    def _invoke_llm(
        self,
        text: str,
        client: Any,
        instructions: Optional[str] = None,
        *args,
        **kwargs,
    ) -> LLMResponse:
        """Wrapper for manifest client.

        To use manifest for guardrailse, do
        ```
        client = Manifest(client_name=..., client_connection=...)
        raw_llm_response, validated_response, *rest = guard(
            client,
            prompt_params={...},
            ...
        ```
        """
        try:
            import manifest  # noqa: F401 # type: ignore
        except ImportError:
            raise PromptCallableException(
                "The `manifest` package is not installed. "
                "Install with `poetry add manifest-ml`"
            )
        client = cast(manifest.Manifest, client)
        manifest_response = client.run(
            nonchat_prompt(prompt=text, instructions=instructions), *args, **kwargs
        )
        return LLMResponse(
            output=manifest_response,
        )



class LiteLLMCallable(PromptCallableBase):
    def _invoke_llm(
        self,
        model: str = "gpt-3.5-turbo",
        messages: Optional[List[Dict]] = None,
        *args,
        **kwargs,
    ) -> LLMResponse:
        """Wrapper for Lite LLM completions.

        To use Lite LLM for guardrails, do
        ```
        from litellm import completion

        raw_llm_response, validated_response = guard(
            completion,
            model="gpt-3.5-turbo",
            prompt_params={...},
            temperature=...,
            ...
        )
        ```
        """
        try:
            from litellm import completion  # type: ignore
        except ImportError as e:
            raise PromptCallableException(
                "The `litellm` package is not installed. "
                "Install with `pip install litellm`"
            ) from e
        
        response = completion(
            model=model,
            messages=messages,
            *args,
            **kwargs,
        )

        if kwargs.get("stream", False):
            # If stream is defined and set to True,
            # the callable returns a generator object
            llm_response = cast(Iterable[str], response)
            return LLMResponse(
                output="",
                stream_output=llm_response,
            )

        return LLMResponse(
            output=response.choices[0].message.content,  # type: ignore
            prompt_token_count=response.usage.prompt_tokens,  # type: ignore
            response_token_count=response.usage.completion_tokens,  # type: ignore
        )


class HuggingFaceModelCallable(PromptCallableBase):
    def _invoke_llm(
        self, prompt: str, model_generate: Any, *args, **kwargs
    ) -> LLMResponse:
        try:
            import transformers  # noqa: F401 # type: ignore
        except ImportError:
            raise PromptCallableException(
                "The `transformers` package is not installed. "
                "Install with `pip install transformers`"
            )
        try:
            import torch
        except ImportError:
            raise PromptCallableException(
                "The `torch` package is not installed. "
                "Install with `pip install torch`"
            )

        tokenizer = kwargs.pop("tokenizer")
        if not tokenizer:
            raise UserFacingException(
                ValueError(
                    "'tokenizer' must be provided in order to use Hugging Face models!"
                )
            )

        torch_device = "cuda" if torch.cuda.is_available() else "cpu"

        return_tensors = kwargs.pop("return_tensors", "pt")
        skip_special_tokens = kwargs.pop("skip_special_tokens", True)

        input_ids = kwargs.pop("input_ids", None)
        input_values = kwargs.pop("input_values", None)
        input_features = kwargs.pop("input_features", None)
        pixel_values = kwargs.pop("pixel_values", None)
        model_inputs = kwargs.pop("model_inputs", {})
        if (
            input_ids is None
            and input_values is None
            and input_features is None
            and pixel_values is None
            and not model_inputs
        ):
            model_inputs = tokenizer(prompt, return_tensors=return_tensors).to(
                torch_device
            )
        else:
            model_inputs["input_ids"] = input_ids
            model_inputs["input_values"] = input_values
            model_inputs["input_features"] = input_features
            model_inputs["pixel_values"] = pixel_values

        do_sample = kwargs.pop("do_sample", None)
        temperature = kwargs.pop("temperature", None)
        if not do_sample and temperature == 0:
            temperature = None

        model_inputs["do_sample"] = do_sample
        model_inputs["temperature"] = temperature

        output = model_generate(
            **model_inputs,
            **kwargs,
        )

        # NOTE: This is currently restricted to single outputs
        # Should we choose to support multiple return sequences,
        # We would need to either validate all of them
        # and choose the one with the least failures,
        # or accept a selection function
        decoded_output = tokenizer.decode(
            output[0], skip_special_tokens=skip_special_tokens
        )

        return LLMResponse(output=decoded_output)


class HuggingFacePipelineCallable(PromptCallableBase):
    def _invoke_llm(self, prompt: str, pipeline: Any, *args, **kwargs) -> LLMResponse:
        try:
            import transformers  # noqa: F401 # type: ignore
        except ImportError:
            raise PromptCallableException(
                "The `transformers` package is not installed. "
                "Install with `pip install transformers`"
            )
        try:
            import torch  # noqa: F401 # type: ignore
        except ImportError:
            raise PromptCallableException(
                "The `torch` package is not installed. "
                "Install with `pip install torch`"
            )

        content_key = kwargs.pop("content_key", "generated_text")

        temperature = kwargs.pop("temperature", None)
        if temperature == 0:
            temperature = None

        output = pipeline(
            prompt,
            temperature=temperature,
            *args,
            **kwargs,
        )

        # NOTE: This is currently restricted to single outputs
        # Should we choose to support multiple return sequences,
        # We would need to either validate all of them
        # and choose the one with the least failures,
        # or accept a selection function
        content = safe_get(output[0], content_key)

        return LLMResponse(output=content)


class ArbitraryCallable(PromptCallableBase):
    def __init__(self, llm_api: Callable, *args, **kwargs):
        self.llm_api = llm_api
        super().__init__(*args, **kwargs)

    def _invoke_llm(self, *args, **kwargs) -> LLMResponse:
        """Wrapper for arbitrary callable.

        To use an arbitrary callable for guardrails, do
        ```
        raw_llm_response, validated_response, *rest = guard(
            my_callable,
            prompt_params={...},
            ...
        )
        ```
        """
        # Get the response from the callable
        # The LLM response should either be a
        # string or an generator object of strings
        llm_response = self.llm_api(*args, **kwargs)

        # Check if kwargs stream is passed in
        if kwargs.get("stream", False):
            # If stream is defined and set to True,
            # the callable returns a generator object
            llm_response = cast(Iterable[str], llm_response)
            return LLMResponse(
                output="",
                stream_output=llm_response,
            )

        # Else, the callable returns a string
        llm_response = cast(str, llm_response)
        return LLMResponse(
            output=llm_response,
        )


def get_llm_ask(llm_api: Callable, *args, **kwargs) -> PromptCallableBase:
    if "temperature" not in kwargs:
        kwargs.update({"temperature": 0})

    try:
        import manifest  # noqa: F401 # type: ignore

        if isinstance(llm_api, manifest.Manifest):
            return ManifestCallable(*args, client=llm_api, **kwargs)
    except ImportError:
        pass

    try:
        from transformers import (  # noqa: F401 # type: ignore
            FlaxPreTrainedModel,
            GenerationMixin,
            PreTrainedModel,
            TFPreTrainedModel,
        )

        api_self = getattr(llm_api, "__self__", None)

        if (
            isinstance(api_self, PreTrainedModel)
            or isinstance(api_self, TFPreTrainedModel)
            or isinstance(api_self, FlaxPreTrainedModel)
        ):
            if (
                hasattr(llm_api, "__func__")
                and llm_api.__func__ == GenerationMixin.generate
            ):
                return HuggingFaceModelCallable(*args, model_generate=llm_api, **kwargs)
            raise ValueError("Only text generation models are supported at this time.")
    except ImportError:
        pass

    try:
        from transformers import Pipeline  # noqa: F401 # type: ignore

        if isinstance(llm_api, Pipeline):
            # Couldn't find a constant for this
            if llm_api.task == "text-generation":
                return HuggingFacePipelineCallable(*args, pipeline=llm_api, **kwargs)
            raise ValueError(
                "Only text generation pipelines are supported at this time."
            )
    except ImportError:
        pass

    try:
        from litellm import completion  # noqa: F401 # type: ignore

        if llm_api == completion or (llm_api == None and kwargs.get("model")):
            return LiteLLMCallable(*args, **kwargs)
    except ImportError:
        pass

    # Let the user pass in an arbitrary callable.
    return ArbitraryCallable(*args, llm_api=llm_api, **kwargs)


###
# Async wrappers
###


class AsyncPromptCallableBase(PromptCallableBase):
    async def invoke_llm(
        self,
        *args,
        **kwargs,
    ) -> LLMResponse:
        raise NotImplementedError

    async def __call__(self, *args, **kwargs) -> LLMResponse:
        try:
            result = await self.invoke_llm(
                *self.init_args, *args, **self.init_kwargs, **kwargs
            )
        except Exception as e:
            raise PromptCallableException(
                "The callable `fn` passed to `Guard(fn, ...)` failed"
                f" with the following error: `{e}`. "
                "Make sure that `fn` can be called as a function that"
                " takes in a single prompt string "
                "and returns a string."
            )
        if not isinstance(result, LLMResponse):
            raise PromptCallableException(
                "The callable `fn` passed to `Guard(fn, ...)` returned"
                f" a non-string value: {result}. "
                "Make sure that `fn` can be called as a function that"
                " takes in a single prompt string "
                "and returns a string."
            )
        return result


class AsyncOpenAIModel(AsyncPromptCallableBase):
    pass


class AsyncOpenAICallable(AsyncOpenAIModel):
    async def invoke_llm(
        self,
        text: str,
        engine: str = "text-davinci-003",
        instructions: Optional[str] = None,
        *args,
        **kwargs,
    ):
        if "api_key" in kwargs:
            api_key = kwargs.pop("api_key")
        else:
            api_key = None

        if "model" in kwargs:
            engine = kwargs.pop("model")

        aclient = AsyncOpenAIClient(api_key=api_key)
        return await aclient.create_completion(
            engine=engine,
            prompt=nonchat_prompt(prompt=text, instructions=instructions),
            *args,
            **kwargs,
        )


class AsyncLiteLLMCallable(AsyncPromptCallableBase):
    async def invoke_llm(
        self,
        messages,
        *args,
        **kwargs,
    ):
        """Wrapper for Lite LLM completions.

        To use Lite LLM for guardrails, do
        ```
        raw_llm_response, validated_response = guard(
            model="gpt-3.5-turbo",
            prompt_params={...},
            temperature=...,
            ...
        )
        ```
        """
        try:
            from litellm import acompletion  # type: ignore
        except ImportError as e:
            raise PromptCallableException(
                "The `litellm` package is not installed. "
                "Install with `pip install litellm`"
            ) from e


        response = await acompletion(
            messages=messages,
            *args,
            **kwargs,
        )
        if kwargs.get("stream", False):
            # If stream is defined and set to True,
            # the callable returns a generator object
            # response = cast(AsyncIterable[str], response)
            return LLMResponse(
                output="",
                async_stream_output=response.completion_stream,  # pyright: ignore[reportGeneralTypeIssues]
            )

        return LLMResponse(
            output=response.choices[0].message.content,  # type: ignore
            prompt_token_count=response.usage.prompt_tokens,  # type: ignore
            response_token_count=response.usage.completion_tokens,  # type: ignore
        )


class AsyncManifestCallable(AsyncPromptCallableBase):
    async def invoke_llm(
        self,
        text: str,
        client: Any,
        instructions: Optional[str] = None,
        *args,
        **kwargs,
    ):
        """Async wrapper for manifest client.

        To use manifest for guardrails, do
        ```
        client = Manifest(client_name=..., client_connection=...)
        raw_llm_response, validated_response, *rest = guard(
            client,
            prompt_params={...},
            ...
        ```
        """
        try:
            import manifest  # noqa: F401 # type: ignore
        except ImportError:
            raise PromptCallableException(
                "The `manifest` package is not installed. "
                "Install with `poetry add manifest-ml`"
            )
        client = cast(manifest.Manifest, client)
        manifest_response = await client.arun_batch(
            prompts=[nonchat_prompt(prompt=text, instructions=instructions)],
            *args,
            **kwargs,
        )
        if kwargs.get("stream", False):
            raise NotImplementedError(
                "Manifest async streaming is not yet supported by manifest."
            )
        return LLMResponse(
            output=manifest_response[0],
        )


class AsyncArbitraryCallable(AsyncPromptCallableBase):
    def __init__(self, llm_api: Callable, *args, **kwargs):
        self.llm_api = llm_api
        super().__init__(*args, **kwargs)

    async def invoke_llm(self, *args, **kwargs) -> LLMResponse:
        """Wrapper for arbitrary callable.

        To use an arbitrary callable for guardrails, do
        ```
        raw_llm_response, validated_response, *rest = guard(
            my_callable,
            prompt_params={...},
            ...
        )
        ```
        """
        output = await self.llm_api(*args, **kwargs)
        if kwargs.get("stream", False):
            # If stream is defined and set to True,
            # the callable returns a generator object
            return LLMResponse(
                output="",
                async_stream_output=output.completion_stream,
            )
        return LLMResponse(
            output=output,
        )


def get_async_llm_ask(
    llm_api: Callable[[Any], Awaitable[Any]], *args, **kwargs
) -> AsyncPromptCallableBase:
    try:
        import manifest  # noqa: F401 # type: ignore

        if isinstance(llm_api, manifest.Manifest):
            return AsyncManifestCallable(*args, client=llm_api, **kwargs)
    except ImportError:
        pass

    try:
        import litellm

        if llm_api == litellm.acompletion or (llm_api == None and kwargs.get("model")):
            return AsyncLiteLLMCallable(*args, **kwargs)
    except ImportError:
        pass

    return AsyncArbitraryCallable(*args, llm_api=llm_api, **kwargs)


def model_is_supported_server_side(
    llm_api: Optional[Union[Callable, Callable[[Any], Awaitable[Any]]]] = None,
    *args,
    **kwargs,
) -> bool:
    if not llm_api:
        return True
    # TODO: Support other models; requires server-side updates
    model = get_llm_ask(llm_api, *args, **kwargs)
    if asyncio.iscoroutinefunction(llm_api):
        model = get_async_llm_ask(llm_api, *args, **kwargs)
    return (
        issubclass(type(model), OpenAIModel)
        or issubclass(type(model), AsyncOpenAIModel)
        or isinstance(model, LiteLLMCallable)
        or isinstance(model, AsyncLiteLLMCallable)
    )


# CONTINUOUS FIXME: Update with newly supported LLMs
def get_llm_api_enum(
    llm_api: Callable[[Any], Awaitable[Any]], *args, **kwargs
) -> Optional[LLMResource]:
    # TODO: Distinguish between v1 and v2
    model = get_llm_ask(llm_api, *args, **kwargs)
    if llm_api == get_static_openai_create_func():
        return LLMResource.OPENAI_DOT_COMPLETION_DOT_CREATE
    elif llm_api == get_static_openai_chat_create_func():
        return LLMResource.OPENAI_DOT_CHAT_COMPLETION_DOT_CREATE
    elif llm_api == get_static_openai_acreate_func():
        return LLMResource.OPENAI_DOT_COMPLETION_DOT_ACREATE
    elif llm_api == get_static_openai_chat_acreate_func():
        return LLMResource.OPENAI_DOT_CHAT_COMPLETION_DOT_ACREATE
    elif isinstance(model, LiteLLMCallable):
        return LLMResource.LITELLM_DOT_COMPLETION
    elif isinstance(model, AsyncLiteLLMCallable):
        return LLMResource.LITELLM_DOT_ACOMPLETION

    else:
        return None
