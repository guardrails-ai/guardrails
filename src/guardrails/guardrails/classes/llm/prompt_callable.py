from guardrails.classes.llm.llm_response import LLMResponse


CALLABLE_FAILURE_SUFFIX = """Make sure that `fn` can be called as a function
that accepts a prompt string, **kwargs, and returns a string.
 If you're using a custom LLM callable, please see docs
 here: https://go.guardrailsai.com/B1igEy3"""  # noqa


class PromptCallableException(Exception):
    pass


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
                f" with the following error: `{e}`. {CALLABLE_FAILURE_SUFFIX}"
            )
        if not isinstance(result, LLMResponse):
            raise PromptCallableException(
                "The callable `fn` passed to `Guard(fn, ...)` returned"
                f" a non-string value: {result}. {CALLABLE_FAILURE_SUFFIX}"
            )
        return result
