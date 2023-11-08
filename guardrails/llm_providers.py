import os
from typing import Any, Awaitable, Callable, Dict, List, Optional, cast

import openai
import openai.error
import tiktoken
from pydantic import BaseModel
from tenacity import retry, retry_if_exception_type, wait_exponential_jitter

from guardrails.utils.logs_utils import LLMResponse
from guardrails.utils.pydantic_utils import convert_pydantic_model_to_openai_fn

OPENAI_RETRYABLE_ERRORS = [
    openai.error.APIConnectionError,
    openai.error.APIError,
    openai.error.TryAgain,
    openai.error.Timeout,
    openai.error.RateLimitError,
    openai.error.ServiceUnavailableError,
]
RETRYABLE_ERRORS = tuple(OPENAI_RETRYABLE_ERRORS)


class PromptCallableException(Exception):
    pass


def num_tokens_from_string(text: str, model_name: str) -> int:
    """Returns the number of tokens in a text string.

    Supported for OpenAI models only. This is a helper function
    that is required when OpenAI's `stream` parameter is set to `True`,
    because OpenAI does not return the number of tokens in that case.
    Requires the `tiktoken` package to be installed.

    Args:
        text (str): The text string to count the number of tokens in.
        model_name (str): The name of the OpenAI model to use.

    Returns:
        num_tokens (int): The number of tokens in the text string.
    """
    encoding = tiktoken.encoding_for_model(model_name)
    num_tokens = len(encoding.encode(text))
    return num_tokens


def num_tokens_from_messages(
    messages: List[Dict[str, str]], model: str = "gpt-3.5-turbo-0613"
) -> int:
    """Return the number of tokens used by a list of messages."""
    try:
        encoding = tiktoken.encoding_for_model(model)
    except KeyError:
        print("Warning: model not found. Using cl100k_base encoding.")
        encoding = tiktoken.get_encoding("cl100k_base")
    if model in {
        "gpt-3.5-turbo-0613",
        "gpt-3.5-turbo-16k-0613",
        "gpt-4-0314",
        "gpt-4-32k-0314",
        "gpt-4-0613",
        "gpt-4-32k-0613",
    }:
        tokens_per_message = 3
        tokens_per_name = 1
    elif model == "gpt-3.5-turbo-0301":
        tokens_per_message = (
            4  # every message follows <|start|>{role/name}\n{content}<|end|>\n
        )
        tokens_per_name = -1  # if there's a name, the role is omitted
    elif "gpt-3.5-turbo" in model:
        print(
            """Warning: gpt-3.5-turbo may update over time.
            Returning num tokens assuming gpt-3.5-turbo-0613."""
        )
        return num_tokens_from_messages(messages, model="gpt-3.5-turbo-0613")
    elif "gpt-4" in model:
        print(
            """Warning: gpt-4 may update over time.
            Returning num tokens assuming gpt-4-0613."""
        )
        return num_tokens_from_messages(messages, model="gpt-4-0613")
    else:
        raise NotImplementedError(
            f"""num_tokens_from_messages() is not implemented for model {model}.
            See https://github.com/openai/openai-python/blob/main/chatml.md for
            information on how messages are converted to tokens."""
        )

    num_tokens = 0
    for message in messages:
        num_tokens += tokens_per_message
        for key, value in message.items():
            num_tokens += len(encoding.encode(value))
            if key == "name":
                num_tokens += tokens_per_name

    # every reply is primed with <|start|>assistant<|message|>
    num_tokens += 3
    return num_tokens


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

    @retry(
        wait=wait_exponential_jitter(max=60),
        retry=retry_if_exception_type(RETRYABLE_ERRORS),
    )
    def _call_llm(self, *args, **kwargs) -> LLMResponse:
        return self._invoke_llm(*self.init_args, *args, **self.init_kwargs, **kwargs)

    def __call__(self, *args, **kwargs) -> LLMResponse:
        try:
            result = self._call_llm(*args, **kwargs)
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
        api_key = kwargs.pop("api_key", os.environ.get("OPENAI_API_KEY"))
        openai_response = openai.Completion.create(
            api_key=api_key,
            engine=engine,
            prompt=nonchat_prompt(prompt=text, instructions=instructions),
            *args,
            **kwargs,
        )

        # Check if kwargs stream is passed in
        if kwargs.get("stream", None) is None:
            # If stream is not defined, return default behavior
            return LLMResponse(
                output=openai_response["choices"][0]["text"],  # type: ignore
                prompt_token_count=openai_response["usage"][  # type: ignore
                    "prompt_tokens"
                ],
                response_token_count=openai_response["usage"][  # type: ignore
                    "completion_tokens"
                ],
            )
        else:
            # If stream is defined, openai returns a generator
            # that we need to iterate through
            complete_output = ""
            for response in openai_response:
                complete_output += response["choices"][0]["text"]

            # Also, it no longer returns usage information
            # So manually count the tokens using tiktoken
            prompt_token_count = num_tokens_from_string(
                text=nonchat_prompt(prompt=text, instructions=instructions),
                model_name=engine,
            )
            response_token_count = num_tokens_from_string(
                text=complete_output, model_name=engine
            )

            # Return the LLMResponse
            return LLMResponse(
                output=complete_output,
                prompt_token_count=prompt_token_count,
                response_token_count=response_token_count,
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
        raw_llm_response, validated_response = guard(
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
        api_key = kwargs.pop("api_key", os.environ.get("OPENAI_API_KEY"))
        openai_response = openai.ChatCompletion.create(
            api_key=api_key,
            model=model,
            messages=chat_prompt(
                prompt=text, instructions=instructions, msg_history=msg_history
            ),
            *args,
            **fn_kwargs,
            **kwargs,
        )

        # Check if kwargs stream is passed in
        if kwargs.get("stream", None) is None:
            # If stream is not defined, return default behavior
            # Extract string from response
            if (
                "function_call" in openai_response["choices"][0]["message"]
            ):  # type: ignore
                output = openai_response["choices"][0]["message"][  # type: ignore
                    "function_call"
                ]["arguments"]
            else:
                output = openai_response["choices"][0]["message"][
                    "content"
                ]  # type: ignore

            return LLMResponse(
                output=output,
                prompt_token_count=openai_response["usage"][  # type: ignore
                    "prompt_tokens"
                ],
                response_token_count=openai_response["usage"][  # type: ignore
                    "completion_tokens"
                ],
            )
        else:
            # If stream is defined, openai returns a generator
            # that we need to iterate through
            collected_messages = []
            # iterate through the stream of events
            for chunk in openai_response:
                chunk_message = chunk["choices"][0]["delta"]  # extract the message
                collected_messages.append(chunk_message)  # save the message

            complete_output = "".join(
                [msg.get("content", "") for msg in collected_messages]
            )

            # Also, it no longer returns usage information
            # So manually count the tokens using tiktoken
            prompt_token_count = num_tokens_from_messages(
                messages=chat_prompt(
                    prompt=text, instructions=instructions, msg_history=msg_history
                ),
                model=model,
            )
            response_token_count = num_tokens_from_string(
                text=complete_output, model_name=model
            )

            # Return the LLMResponse
            return LLMResponse(
                output=complete_output,
                prompt_token_count=prompt_token_count,
                response_token_count=response_token_count,
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
        raw_llm_response, validated_response = guard(
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
                "Install with `pip install manifest-ml`"
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

        raw_llm_response, validated_response = guard(
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


class ArbitraryCallable(PromptCallableBase):
    def __init__(self, llm_api: Callable, *args, **kwargs):
        self.llm_api = llm_api
        super().__init__(*args, **kwargs)

    def _invoke_llm(self, *args, **kwargs) -> LLMResponse:
        """Wrapper for arbitrary callable.

        To use an arbitrary callable for guardrails, do
        ```
        raw_llm_response, validated_response = guard(
            my_callable,
            prompt_params={...},
            ...
        )
        ```
        """
        return LLMResponse(
            output=self.llm_api(*args, **kwargs),
        )


def get_llm_ask(llm_api: Callable, *args, **kwargs) -> PromptCallableBase:
    if "temperature" not in kwargs:
        kwargs.update({"temperature": 0})
    if llm_api == openai.Completion.create:
        return OpenAICallable(*args, **kwargs)
    if llm_api == openai.ChatCompletion.create:
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

    @retry(
        wait=wait_exponential_jitter(max=60),
        retry=retry_if_exception_type(RETRYABLE_ERRORS),
    )
    async def call_llm(self, *args, **kwargs) -> LLMResponse:
        return await self.invoke_llm(
            *self.init_args, *args, **self.init_kwargs, **kwargs
        )

    async def __call__(self, *args, **kwargs) -> LLMResponse:
        try:
            result = await self.call_llm(*args, **kwargs)
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
        api_key = kwargs.pop("api_key", os.environ.get("OPENAI_API_KEY"))
        openai_response = await openai.Completion.acreate(
            api_key=api_key,
            engine=engine,
            prompt=nonchat_prompt(prompt=text, instructions=instructions),
            *args,
            **kwargs,
        )
        return LLMResponse(
            output=openai_response["choices"][0]["text"],  # type: ignore
            prompt_token_count=openai_response["usage"][  # type: ignore
                "prompt_tokens"
            ],
            response_token_count=openai_response["usage"][  # type: ignore
                "completion_tokens"
            ],
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
        raw_llm_response, validated_response = guard(
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
        api_key = kwargs.pop("api_key", os.environ.get("OPENAI_API_KEY"))
        openai_response = await openai.ChatCompletion.acreate(
            api_key=api_key,
            model=model,
            messages=chat_prompt(
                prompt=text, instructions=instructions, msg_history=msg_history
            ),
            *args,
            **fn_kwargs,
            **kwargs,
        )

        # Extract string from response
        if "function_call" in openai_response["choices"][0]["message"]:  # type: ignore
            output = openai_response["choices"][0]["message"][  # type: ignore
                "function_call"
            ]["arguments"]
        else:
            output = openai_response["choices"][0]["message"]["content"]  # type: ignore

        return LLMResponse(
            output=output,
            prompt_token_count=openai_response["usage"][  # type: ignore
                "prompt_tokens"
            ],
            response_token_count=openai_response["usage"][  # type: ignore
                "completion_tokens"
            ],
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
        raw_llm_response, validated_response = guard(
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
                "Install with `pip install manifest-ml`"
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
        raw_llm_response, validated_response = guard(
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
    if llm_api == openai.Completion.acreate:
        return AsyncOpenAICallable(*args, **kwargs)
    if llm_api == openai.ChatCompletion.acreate:
        return AsyncOpenAIChatCallable(*args, **kwargs)

    try:
        import manifest  # noqa: F401 # type: ignore

        if isinstance(llm_api, manifest.Manifest):
            return AsyncManifestCallable(*args, client=llm_api, **kwargs)
    except ImportError:
        pass

    return AsyncArbitraryCallable(*args, llm_api=llm_api, **kwargs)
