from guardrails.utils.openai_utils import OpenAIServiceUnavailableError


class MockOpenAILlm:
    def __init__(self, times_called=0, response="Hello world!"):
        self.times_called = times_called
        self.response = response

    def fail_retryable(self, prompt: str, *args, **kwargs) -> str:
        if self.times_called == 0:
            self.times_called = self.times_called + 1
            raise OpenAIServiceUnavailableError("ServiceUnavailableError")
        return self.response

    def fail_non_retryable(self, prompt: str, *args, **kwargs) -> str:
        raise Exception("Non-Retryable Error!")

    def succeed(self, prompt: str, *args, **kwargs) -> str:
        return self.response


class MockAsyncOpenAILlm:
    def __init__(self, times_called=0, response="Hello world!"):
        self.times_called = times_called
        self.response = response

    async def fail_retryable(self, prompt: str, *args, **kwargs) -> str:
        if self.times_called == 0:
            self.times_called = self.times_called + 1
            raise OpenAIServiceUnavailableError("ServiceUnavailableError")
        return self.response

    async def fail_non_retryable(self, prompt: str, *args, **kwargs) -> str:
        raise Exception("Non-Retryable Error!")

    async def succeed(self, prompt: str, *args, **kwargs) -> str:
        return self.response
