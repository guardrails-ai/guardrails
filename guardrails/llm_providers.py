from typing import Any, Awaitable, Callable, Dict, Iterable, List, Optional, cast

from pydantic import BaseModel

from guardrails.utils.exception_utils import UserFacingException
from guardrails.utils.llm_response import LLMResponse
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


def nonchat_prompt(prompt: str, instructions: Optional[str] = None) -> str:
    """Prepare final prompt for nonchat engine."""
    if instructions:
        prompt = "\n\n".join([instructions, prompt])
    return prompt


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


class OpenAICallable(PromptCallableBase):
    def _invoke_llm(
        self,
        text: str,
        engine: str = "text-davinci-003",
        instructions: Optional[str] = None,
        *args,
        **kwargs,
    ) -> LLMResponse:
        if "api_key" in kwargs:
            api_key = kwargs.pop("api_key")
        else:
            api_key = None

        if "model" in kwargs:
            engine = kwargs.pop("model")

        client = OpenAIClient(api_key=api_key)
        return client.create_completion(
            engine=engine,
            prompt=nonchat_prompt(prompt=text, instructions=instructions),
            *args,
            **kwargs,
        )


class OpenAIChatCallable(PromptCallableBase):
    def _invoke_llm(
        self,
        text: Optional[str] = None,
        model: str = "gpt-3.5-turbo",
        instructions: Optional[str] = None,
        msg_history: Optional[List[Dict]] = None,
        base_model: Optional[BaseModel] = None,
        function_call: Optional[Any] = None,
        *args,
        **kwargs,
    ) -> LLMResponse:
        """Wrapper for OpenAI chat engines.

        Use Guardrails with OpenAI chat engines by doing
        ```
        raw_llm_response, validated_response, *rest = guard(
            openai.ChatCompletion.create,
            prompt_params={...},
            text=...,
            instructions=...,
            msg_history=...,
            temperature=...,
            ...
        )
        ```

        If `base_model` is passed, the chat engine will be used as a function
        on the base model.
        """

        if msg_history is None and text is None:
            raise PromptCallableException(
                "You must pass in either `text` or `msg_history` to `guard.__call__`."
            )

        # Configure function calling if applicable (only for non-streaming)
        if base_model and not kwargs.get("stream", False):
            function_params = [convert_pydantic_model_to_openai_fn(base_model)]
            if function_call is None:
                function_call = {"name": function_params[0]["name"]}
            fn_kwargs = {"functions": function_params, "function_call": function_call}
        else:
            fn_kwargs = {}

        # Call OpenAI
        if "api_key" in kwargs:
            api_key = kwargs.pop("api_key")
        else:
            api_key = None

        client = OpenAIClient(api_key=api_key)
        return client.create_chat_completion(
            model=model,
            messages=chat_prompt(
                prompt=text, instructions=instructions, msg_history=msg_history
            ),
            *args,
            **fn_kwargs,
            **kwargs,
        )


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


class CohereCallable(PromptCallableBase):
    def _invoke_llm(
        self, prompt: str, client_callable: Any, model: str, *args, **kwargs
    ) -> LLMResponse:
        """To use cohere for guardrails, do ``` client =
        cohere.Client(api_key=...)

        raw_llm_response, validated_response, *rest = guard(
            client.generate,
            prompt_params={...},
            model="command-nightly",
            ...
        )
        ```
        """  # noqa

        if "instructions" in kwargs:
            prompt = kwargs.pop("instructions") + "\n\n" + prompt

        cohere_response = client_callable(prompt=prompt, model=model, *args, **kwargs)
        return LLMResponse(
            output=cohere_response[0].text,
        )


class AnthropicCallable(PromptCallableBase):
    def _invoke_llm(
        self,
        prompt: str,
        client_callable: Any,
        model: str = "claude-instant-1",
        max_tokens_to_sample: int = 100,
        *args,
        **kwargs,
    ) -> LLMResponse:
        """Wrapper for Anthropic Completions.

        To use Anthropic for guardrails, do
        ```
        client = anthropic.Anthropic(api_key=...)

        raw_llm_response, validated_response = guard(
            client,
            model="claude-2",
            max_tokens_to_sample=200,
            prompt_params={...},
            ...
        ```
        """
        try:
            import anthropic
        except ImportError:
            raise PromptCallableException(
                "The `anthropic` package is not installed. "
                "Install with `pip install anthropic`"
            )

        if "instructions" in kwargs:
            prompt = kwargs.pop("instructions") + "\n\n" + prompt

        anthropic_prompt = f"{anthropic.HUMAN_PROMPT} {prompt} {anthropic.AI_PROMPT}"

        anthropic_response = client_callable(
            model=model,
            prompt=anthropic_prompt,
            max_tokens_to_sample=max_tokens_to_sample,
            *args,
            **kwargs,
        )
        return LLMResponse(output=anthropic_response.completion)


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
    if llm_api == get_static_openai_create_func():
        return OpenAICallable(*args, **kwargs)
    if llm_api == get_static_openai_chat_create_func():
        return OpenAIChatCallable(*args, **kwargs)

    try:
        import manifest  # noqa: F401 # type: ignore

        if isinstance(llm_api, manifest.Manifest):
            return ManifestCallable(*args, client=llm_api, **kwargs)
    except ImportError:
        pass

    try:
        import cohere  # noqa: F401 # type: ignore

        if (
            isinstance(getattr(llm_api, "__self__", None), cohere.Client)
            and getattr(llm_api, "__name__", None) == "generate"
        ):
            return CohereCallable(*args, client_callable=llm_api, **kwargs)
    except ImportError:
        pass

    try:
        import anthropic.resources  # noqa: F401 # type: ignore

        if isinstance(
            getattr(llm_api, "__self__", None),
            anthropic.resources.completions.Completions,
        ):
            return AnthropicCallable(*args, client_callable=llm_api, **kwargs)
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


class AsyncOpenAICallable(AsyncPromptCallableBase):
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


class AsyncOpenAIChatCallable(AsyncPromptCallableBase):
    async def invoke_llm(
        self,
        text: Optional[str] = None,
        model: str = "gpt-3.5-turbo",
        instructions: Optional[str] = None,
        msg_history: Optional[List[Dict]] = None,
        base_model: Optional[BaseModel] = None,
        function_call: Optional[Any] = None,
        *args,
        **kwargs,
    ) -> LLMResponse:
        """Wrapper for OpenAI chat engines.

        Use Guardrails with OpenAI chat engines by doing
        ```
        raw_llm_response, validated_response, *rest = guard(
            openai.ChatCompletion.create,
            prompt_params={...},
            text=...,
            instructions=...,
            msg_history=...,
            temperature=...,
            ...
        )
        ```

        If `base_model` is passed, the chat engine will be used as a function
        on the base model.
        """

        if msg_history is None and text is None:
            raise PromptCallableException(
                "You must pass in either `text` or `msg_history` to `guard.__call__`."
            )

        # Configure function calling if applicable
        if base_model:
            function_params = [convert_pydantic_model_to_openai_fn(base_model)]
            if function_call is None:
                function_call = {"name": function_params[0]["name"]}
            fn_kwargs = {"functions": function_params, "function_call": function_call}
        else:
            fn_kwargs = {}

        # Call OpenAI
        if "api_key" in kwargs:
            api_key = kwargs.pop("api_key")
        else:
            api_key = None

        aclient = AsyncOpenAIClient(api_key=api_key)
        return await aclient.create_chat_completion(
            model=model,
            messages=chat_prompt(
                prompt=text, instructions=instructions, msg_history=msg_history
            ),
            *args,
            **fn_kwargs,
            **kwargs,
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
        return LLMResponse(
            output=output,
        )


def get_async_llm_ask(
    llm_api: Callable[[Any], Awaitable[Any]], *args, **kwargs
) -> AsyncPromptCallableBase:
    # these only work with openai v0 (None otherwise)
    if llm_api == get_static_openai_acreate_func():
        return AsyncOpenAICallable(*args, **kwargs)
    if llm_api == get_static_openai_chat_acreate_func():
        return AsyncOpenAIChatCallable(*args, **kwargs)

    try:
        import manifest  # noqa: F401 # type: ignore

        if isinstance(llm_api, manifest.Manifest):
            return AsyncManifestCallable(*args, client=llm_api, **kwargs)
    except ImportError:
        pass

    return AsyncArbitraryCallable(*args, llm_api=llm_api, **kwargs)
