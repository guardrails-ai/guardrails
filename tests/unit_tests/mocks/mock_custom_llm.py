from unittest.mock import Mock
from openai import APIError


class MockOpenAILlm:
    def __init__(self, times_called=0, response="Hello world!"):
        self.times_called = times_called
        self.response = response

    def fail_retryable(self, messages, *args, **kwargs) -> str:
        if self.times_called == 0:
            self.times_called = self.times_called + 1
            raise APIError("ServiceUnavailableError", Mock(), body=None)
        return self.response

    def fail_non_retryable(self, messages, *args, **kwargs) -> str:
        raise Exception("Non-Retryable Error!")

    def succeed(self, messages, *args, **kwargs) -> str:
        return self.response


class MockAsyncOpenAILlm:
    def __init__(self, times_called=0, response="Hello world!"):
        self.times_called = times_called
        self.response = response

    async def fail_retryable(self, messages, *args, **kwargs) -> str:
        if self.times_called == 0:
            self.times_called = self.times_called + 1
            raise APIError("ServiceUnavailableError", Mock(), body=None)
        return self.response

    async def fail_non_retryable(self, messages, *args, **kwargs) -> str:
        raise Exception("Non-Retryable Error!")

    async def succeed(self, messages, *args, **kwargs) -> str:
        return self.response
