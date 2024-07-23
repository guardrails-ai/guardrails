import asyncio

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
from guardrails.utils.telemetry_utils import trace_llm_call, trace_operation


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


def litellm_messages(
    prompt: Optional[str],
    instructions: Optional[str] = None,
    msg_history: Optional[List[Dict]] = None,
) -> List[Dict[str, str]]:
    """Prepare messages for LiteLLM."""
    if msg_history:
        return msg_history
    if prompt is None:
        raise PromptCallableException(
            "Either `text` or `msg_history` required for `guard.__call__`."
        )

    if instructions:
        prompt = "\n\n".join([instructions, prompt])

    return [{"role": "user", "content": prompt}]


class OpenAIModel(PromptCallableBase):
    pass


class OpenAICallable(OpenAIModel):
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
        trace_llm_call(
            input_messages=[
                {"role": "system", "content": instructions},
                {"role": "user", "content": text},
            ],
            model_name=engine,
        )
        return client.create_completion(
            engine=engine,
            prompt=nonchat_prompt(prompt=text, instructions=instructions),
            *args,
            **kwargs,
        )


class OpenAIChatCallable(OpenAIModel):
    supports_base_model = True

    def _invoke_llm(
        self,
        text: Optional[str] = None,
        model: str = "gpt-3.5-turbo",
        instructions: Optional[str] = None,
        msg_history: Optional[List[Dict]] = None,
        base_model: Optional[
            Union[Type[BaseModel], Type[List[Type[BaseModel]]]]
        ] = None,
        function_call: Optional[Any] = None,
        *args,
        **kwargs,
    ) -> LLMResponse:
        """Wrapper for OpenAI chat engines.

        Use Guardrails with OpenAI chat engines by doing
        ```
        raw_llm_response, validated_response, *rest = guard(
            openai.chat.completions.create,
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

        # TODO: Update this to tools
        # Configure function calling if applicable (only for non-streaming)
        fn_kwargs = {}
        if (
            base_model
            and not kwargs.get("stream", False)
            and not kwargs.get("tools", False)
        ):
            function_params = convert_pydantic_model_to_openai_fn(base_model)
            if function_call is None and function_params:
                function_call = {"name": function_params["name"]}
                fn_kwargs = {
                    "functions": [function_params],
                    "function_call": function_call,
                }

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
        prompt = nonchat_prompt(prompt=text, instructions=instructions)
        trace_operation(
            input_mime_type="application/json",
            input_value={
                **kwargs,
                "prompt": prompt,
                "args": args,
            },
        )

        trace_llm_call(
            input_messages=chat_prompt(text, instructions),
            invocation_parameters={
                **kwargs,
                "prompt": prompt,
            },
        )
        manifest_response = client.run(prompt, *args, **kwargs)
        trace_operation(
            output_mime_type="application/json", output_value=manifest_response
        )
        trace_llm_call(
            output_messages=[{"role": "assistant", "content": manifest_response}]
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

        trace_input_messages = chat_prompt(prompt, kwargs.get("instructions"))
        if "instructions" in kwargs:
            prompt = kwargs.pop("instructions") + "\n\n" + prompt

        def is_base_cohere_chat(func):
            try:
                return (
                    func.__closure__[1].cell_contents.__func__.__qualname__
                    == "BaseCohere.chat"
                )
            except (AttributeError, IndexError):
                return False

        # TODO: When cohere totally gets rid of `generate`,
        #       remove this cond and the final return
        if is_base_cohere_chat(client_callable):
            trace_operation(
                input_mime_type="application/json",
                input_value={**kwargs, "message": prompt, "args": args, "model": model},
            )

            trace_llm_call(
                input_messages=trace_input_messages,
                invocation_parameters={**kwargs, "message": prompt, "model": model},
            )
            cohere_response = client_callable(
                message=prompt, model=model, *args, **kwargs
            )
            trace_operation(
                output_mime_type="application/json", output_value=cohere_response
            )
            trace_llm_call(
                output_messages=[{"role": "assistant", "content": cohere_response.text}]
            )
            return LLMResponse(
                output=cohere_response.text,
            )

        trace_operation(
            input_mime_type="application/json",
            input_value={**kwargs, "prompt": prompt, "args": args, "model": model},
        )

        trace_llm_call(
            input_messages=trace_input_messages,
            invocation_parameters={**kwargs, "prompt": prompt, "model": model},
        )
        cohere_response = client_callable(prompt=prompt, model=model, *args, **kwargs)
        trace_operation(
            output_mime_type="application/json", output_value=cohere_response
        )
        trace_llm_call(
            output_messages=[{"role": "assistant", "content": cohere_response[0].text}]
        )
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

        trace_input_messages = chat_prompt(prompt, kwargs.get("instructions"))
        if "instructions" in kwargs:
            prompt = kwargs.pop("instructions") + "\n\n" + prompt

        anthropic_prompt = f"{anthropic.HUMAN_PROMPT} {prompt} {anthropic.AI_PROMPT}"

        trace_operation(
            input_mime_type="application/json",
            input_value={
                **kwargs,
                "model": model,
                "prompt": anthropic_prompt,
                "max_tokens_to_sample": max_tokens_to_sample,
                "args": args,
            },
        )

        trace_llm_call(
            input_messages=trace_input_messages,
            invocation_parameters={
                **kwargs,
                "model": model,
                "prompt": anthropic_prompt,
                "max_tokens_to_sample": max_tokens_to_sample,
            },
        )

        anthropic_response = client_callable(
            model=model,
            prompt=anthropic_prompt,
            max_tokens_to_sample=max_tokens_to_sample,
            *args,
            **kwargs,
        )
        trace_operation(
            output_mime_type="application/json", output_value=anthropic_response
        )
        trace_llm_call(
            output_messages=[
                {"role": "assistant", "content": anthropic_response.completion}
            ]
        )
        return LLMResponse(output=anthropic_response.completion)


class LiteLLMCallable(PromptCallableBase):
    def _invoke_llm(
        self,
        text: Optional[str] = None,
        model: str = "gpt-3.5-turbo",
        instructions: Optional[str] = None,
        msg_history: Optional[List[Dict]] = None,
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
        if text is not None or instructions is not None or msg_history is not None:
            messages = litellm_messages(
                prompt=text, instructions=instructions, msg_history=msg_history
            )
            kwargs["messages"] = messages

        trace_operation(
            input_mime_type="application/json",
            input_value={
                **kwargs,
                "model": model,
                "args": args,
            },
        )

        function_calling_tools = [
            tool.get("function")
            for tool in kwargs.get("tools", [])
            if isinstance(tool, Dict) and tool.get("type") == "function"
        ]
        trace_llm_call(
            input_messages=kwargs.get("messages"),
            invocation_parameters={
                **kwargs,
                "model": model,
            },
            function_call=kwargs.get(
                "function_call", safe_get(function_calling_tools, 0)
            ),
        )

        response = completion(
            model=model,
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

        trace_operation(output_mime_type="application/json", output_value=response)
        if response.choices[0].message.content is not None:  # type: ignore
            output = response.choices[0].message.content  # type: ignore
        else:
            try:
                output = response.choices[0].message.function_call.arguments  # type: ignore
            except AttributeError:
                try:
                    choice = response.choices[0]  # type: ignore
                    output = choice.message.tool_calls[-1].function.arguments  # type: ignore
                except AttributeError as ae_tools:
                    raise ValueError(
                        "No message content or function"
                        " call arguments returned from OpenAI"
                    ) from ae_tools

        completion_tokens = response.usage.completion_tokens  # type: ignore
        prompt_tokens = response.usage.prompt_tokens  # type: ignore
        total_tokens = None
        if completion_tokens or prompt_tokens:
            total_tokens = (completion_tokens or 0) + (prompt_tokens or 0)

        trace_llm_call(
            output_messages=[choice.message for choice in response.choices],  # type: ignore
            token_count_completion=completion_tokens,  # type: ignore
            token_count_prompt=prompt_tokens,  # type: ignore
            token_count_total=total_tokens,  # type: ignore
        )
        return LLMResponse(
            output=output,  # type: ignore
            prompt_token_count=prompt_tokens,  # type: ignore
            response_token_count=completion_tokens,  # type: ignore
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

        trace_operation(
            input_mime_type="application/json",
            input_value={
                **model_inputs,
                **kwargs,
            },
        )

        trace_llm_call(
            input_messages=chat_prompt(prompt, kwargs.get("instructions")),
            invocation_parameters={
                **model_inputs,
                **kwargs,
            },
        )

        output = model_generate(
            **model_inputs,
            **kwargs,
        )

        trace_operation(output_mime_type="application/json", output_value=output)

        # NOTE: This is currently restricted to single outputs
        # Should we choose to support multiple return sequences,
        # We would need to either validate all of them
        # and choose the one with the least failures,
        # or accept a selection function
        decoded_output = tokenizer.decode(
            output[0], skip_special_tokens=skip_special_tokens
        )

        trace_llm_call(
            output_messages=[{"role": "assistant", "content": decoded_output}]
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

        trace_operation(
            input_mime_type="application/json",
            input_value={
                **kwargs,
                "prompt": prompt,
                "temperature": temperature,
                "args": args,
            },
        )

        trace_llm_call(
            input_messages=chat_prompt(prompt, kwargs.get("instructions")),
            invocation_parameters={
                **kwargs,
                "prompt": prompt,
                "temperature": temperature,
            },
        )

        output = pipeline(
            prompt,
            temperature=temperature,
            *args,
            **kwargs,
        )

        trace_operation(output_mime_type="application/json", output_value=output)

        # NOTE: This is currently restricted to single outputs
        # Should we choose to support multiple return sequences,
        # We would need to either validate all of them
        # and choose the one with the least failures,
        # or accept a selection function
        content = safe_get(output[0], content_key)

        trace_llm_call(output_messages=[{"role": "assistant", "content": content}])

        return LLMResponse(output=content)


class ArbitraryCallable(PromptCallableBase):
    def __init__(self, llm_api: Optional[Callable] = None, *args, **kwargs):
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

        trace_operation(
            input_mime_type="application/json",
            input_value={
                **kwargs,
                "args": args,
            },
        )

        trace_llm_call(
            input_messages=chat_prompt(
                kwargs.get("prompt", ""), kwargs.get("instructions")
            ),
            invocation_parameters={
                **kwargs,
            },
        )

        # Get the response from the callable
        # The LLM response should either be a
        # string or an generator object of strings
        llm_response = self.llm_api(*args, **kwargs)  # type: ignore

        # Check if kwargs stream is passed in
        if kwargs.get("stream", False):
            # If stream is defined and set to True,
            # the callable returns a generator object
            llm_response = cast(Iterable[str], llm_response)
            return LLMResponse(
                output="",
                stream_output=llm_response,
            )

        trace_operation(output_mime_type="application/json", output_value=llm_response)
        trace_llm_call(output_messages=[{"role": "assistant", "content": llm_response}])
        # Else, the callable returns a string
        llm_response = cast(str, llm_response)
        return LLMResponse(
            output=llm_response,
        )


def get_llm_ask(
    llm_api: Optional[Callable] = None,
    *args,
    **kwargs,
) -> Optional[PromptCallableBase]:
    if "temperature" not in kwargs:
        kwargs.update({"temperature": 0})

    try:
        from litellm import completion

        if llm_api == completion or (llm_api is None and kwargs.get("model")):
            return LiteLLMCallable(*args, **kwargs)
    except ImportError:
        pass

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
        ) or getattr(llm_api, "__module__", None) == "cohere.client":
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
                and llm_api.__func__ == GenerationMixin.generate  # type: ignore
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
    if llm_api is not None:
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


class AsyncOpenAIChatCallable(AsyncOpenAIModel):
    supports_base_model = True

    async def invoke_llm(
        self,
        text: Optional[str] = None,
        model: str = "gpt-3.5-turbo",
        instructions: Optional[str] = None,
        msg_history: Optional[List[Dict]] = None,
        base_model: Optional[
            Union[Type[BaseModel], Type[List[Type[BaseModel]]]]
        ] = None,
        function_call: Optional[Any] = None,
        *args,
        **kwargs,
    ) -> LLMResponse:
        """Wrapper for OpenAI chat engines.

        Use Guardrails with OpenAI chat engines by doing
        ```
        raw_llm_response, validated_response, *rest = guard(
            openai.chat.completions.create,
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

        # TODO: Update this to tools
        # Configure function calling if applicable
        fn_kwargs = {}
        kwargs_tools = kwargs.get("tools", False)
        if base_model:
            function_params = convert_pydantic_model_to_openai_fn(base_model)
            if function_call is None and function_params and not kwargs_tools:
                function_call = {"name": function_params["name"]}
                fn_kwargs = {
                    "functions": [function_params],
                    "function_call": function_call,
                }

        # Call OpenAI
        if "api_key" in kwargs:
            api_key = kwargs.pop("api_key")
        else:
            api_key = None

        aclient = AsyncOpenAIClient(api_key=api_key)
        # FIXME: OpenAI async streaming seems to be broken
        return await aclient.create_chat_completion(
            model=model,
            messages=chat_prompt(
                prompt=text, instructions=instructions, msg_history=msg_history
            ),
            *args,
            **fn_kwargs,
            **kwargs,
        )


class AsyncLiteLLMCallable(AsyncPromptCallableBase):
    async def invoke_llm(
        self,
        text: Optional[str] = None,
        instructions: Optional[str] = None,
        msg_history: Optional[List[Dict]] = None,
        *args,
        **kwargs,
    ):
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
            from litellm import acompletion  # type: ignore
        except ImportError as e:
            raise PromptCallableException(
                "The `litellm` package is not installed. "
                "Install with `pip install litellm`"
            ) from e

        if text is not None or instructions is not None or msg_history is not None:
            messages = litellm_messages(
                prompt=text,
                instructions=instructions,
                msg_history=msg_history,
            )
            kwargs["messages"] = messages

        trace_operation(
            input_mime_type="application/json",
            input_value={
                **kwargs,
                "args": args,
            },
        )

        function_calling_tools = [
            tool.get("function")
            for tool in kwargs.get("tools", [])
            if isinstance(tool, Dict) and tool.get("type") == "function"
        ]
        trace_llm_call(
            input_messages=kwargs.get("messages"),
            invocation_parameters={**kwargs},
            function_call=kwargs.get(
                "function_call", safe_get(function_calling_tools, 0)
            ),
        )

        response = await acompletion(
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

        trace_operation(output_mime_type="application/json", output_value=response)
        if response.choices[0].message.content is not None:  # type: ignore
            output = response.choices[0].message.content  # type: ignore
        else:
            try:
                output = response.choices[0].message.function_call.arguments  # type: ignore
            except AttributeError:
                try:
                    choice = response.choices[0]  # type: ignore
                    output = choice.message.tool_calls[-1].function.arguments  # type: ignore
                except AttributeError as ae_tools:
                    raise ValueError(
                        "No message content or function"
                        " call arguments returned from OpenAI"
                    ) from ae_tools

        completion_tokens = response.usage.completion_tokens  # type: ignore
        prompt_tokens = response.usage.prompt_tokens  # type: ignore
        total_tokens = None
        if completion_tokens or prompt_tokens:
            total_tokens = (completion_tokens or 0) + (prompt_tokens or 0)
        trace_llm_call(
            output_messages=[choice.message for choice in response.choices],  # type: ignore
            token_count_completion=completion_tokens,  # type: ignore
            token_count_prompt=prompt_tokens,  # type: ignore
            token_count_total=total_tokens,  # type: ignore
        )
        return LLMResponse(
            output=output,  # type: ignore
            prompt_token_count=prompt_tokens,  # type: ignore
            response_token_count=completion_tokens,  # type: ignore
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

        prompts = [nonchat_prompt(prompt=text, instructions=instructions)]

        trace_operation(
            input_mime_type="application/json",
            input_value={
                **kwargs,
                "prompts": prompts,
                "args": args,
            },
        )

        trace_llm_call(
            input_messages=chat_prompt(text, instructions),
            invocation_parameters={
                **kwargs,
                "prompts": prompts,
            },
        )

        client = cast(manifest.Manifest, client)
        manifest_response = await client.arun_batch(
            prompts=prompts,
            *args,
            **kwargs,
        )
        if kwargs.get("stream", False):
            raise NotImplementedError(
                "Manifest async streaming is not yet supported by manifest."
            )
        trace_operation(
            output_mime_type="application/json", output_value=manifest_response
        )
        trace_llm_call(
            output_messages=[{"role": "assistant", "content": manifest_response[0]}]
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

        trace_operation(
            input_mime_type="application/json",
            input_value={
                **kwargs,
                "args": args,
            },
        )

        trace_llm_call(
            input_messages=chat_prompt(
                kwargs.get("prompt", ""), kwargs.get("instructions")
            ),
            invocation_parameters={
                **kwargs,
            },
        )

        output = await self.llm_api(*args, **kwargs)
        if kwargs.get("stream", False):
            # If stream is defined and set to True,
            # the callable returns a generator object
            return LLMResponse(
                output="",
                async_stream_output=output.completion_stream,
            )

        trace_operation(output_mime_type="application/json", output_value=output)
        trace_llm_call(output_messages=[{"role": "assistant", "content": output}])

        return LLMResponse(
            output=output,
        )


def get_async_llm_ask(
    llm_api: Callable[[Any], Awaitable[Any]], *args, **kwargs
) -> AsyncPromptCallableBase:
    try:
        import litellm

        if llm_api == litellm.acompletion or (llm_api is None and kwargs.get("model")):
            return AsyncLiteLLMCallable(*args, **kwargs)
    except ImportError:
        pass

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
