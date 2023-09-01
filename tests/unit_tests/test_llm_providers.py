import pytest
from guardrails.llm_providers import ArbitraryCallable, AsyncArbitraryCallable, LLMResponse, PromptCallableException
from .mocks import MockCustomLlm, MockAsyncCustomLlm


def test_arbitrary_callable_retries_on_retryable_errors(mocker):
    llm = MockCustomLlm()
    fail_retryable_spy = mocker.spy(llm, 'fail_retryable')
    
    arbitrary_callable = ArbitraryCallable(llm.fail_retryable, prompt="Hello")
    response = arbitrary_callable()

    assert fail_retryable_spy.call_count == 2
    assert isinstance(response, LLMResponse) is True
    assert response.output == "Hello world!"
    assert response.prompt_token_count == None
    assert response.response_token_count == None


def test_arbitrary_callable_does_not_retry_on_non_retryable_errors(mocker):
    with pytest.raises(Exception) as e:
        llm = MockCustomLlm()
        fail_non_retryable_spy = mocker.spy(llm, 'fail_non_retryable')
        
        arbitrary_callable = ArbitraryCallable(llm.fail_retryable, prompt="Hello")
        arbitrary_callable()

        assert fail_non_retryable_spy.call_count == 1
        assert isinstance(e, PromptCallableException) is True
        assert str(e) == "The callable `fn` passed to `Guard(fn, ...)` failed with the following error: `Non-Retryable Error!`. Make sure that `fn` can be called as a function that takes in a single prompt string and returns a string."


def test_arbitrary_callable_does_not_retry_on_success(mocker):
    llm = MockCustomLlm()
    succeed_spy = mocker.spy(llm, 'succeed')
    
    arbitrary_callable = ArbitraryCallable(llm.succeed, prompt="Hello")
    response = arbitrary_callable()

    assert succeed_spy.call_count == 1
    assert isinstance(response, LLMResponse) is True
    assert response.output == "Hello world!"
    assert response.prompt_token_count == None
    assert response.response_token_count == None


@pytest.mark.asyncio
async def test_async_arbitrary_callable_retries_on_retryable_errors(mocker):
    llm = MockAsyncCustomLlm()
    fail_retryable_spy = mocker.spy(llm, 'fail_retryable')
    
    arbitrary_callable = AsyncArbitraryCallable(llm.fail_retryable, prompt="Hello")
    response = await arbitrary_callable()

    assert fail_retryable_spy.call_count == 2
    assert isinstance(response, LLMResponse) is True
    assert response.output == "Hello world!"
    assert response.prompt_token_count == None
    assert response.response_token_count == None


# Passing
@pytest.mark.asyncio
async def test_async_arbitrary_callable_does_not_retry_on_non_retryable_errors(mocker):
    with pytest.raises(Exception) as e:
        llm = MockAsyncCustomLlm()
        fail_non_retryable_spy = mocker.spy(llm, 'fail_non_retryable')
        
        arbitrary_callable = AsyncArbitraryCallable(llm.fail_retryable, prompt="Hello")
        await arbitrary_callable()

        assert fail_non_retryable_spy.call_count == 1
        assert isinstance(e, PromptCallableException) is True
        assert str(e) == "The callable `fn` passed to `Guard(fn, ...)` failed with the following error: `Non-Retryable Error!`. Make sure that `fn` can be called as a function that takes in a single prompt string and returns a string."


@pytest.mark.asyncio
async def test_async_arbitrary_callable_does_not_retry_on_success(mocker):
    llm = MockAsyncCustomLlm()
    succeed_spy = mocker.spy(llm, 'succeed')
    
    arbitrary_callable = AsyncArbitraryCallable(llm.succeed, prompt="Hello")
    response = await arbitrary_callable()

    assert succeed_spy.call_count == 1
    assert isinstance(response, LLMResponse) is True
    assert response.output == "Hello world!"
    assert response.prompt_token_count == None
    assert response.response_token_count == None

