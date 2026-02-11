import os
from unittest.mock import Mock, patch

import pytest

from guardrails.classes.llm.llm_response import LLMResponse
from guardrails.utils.openai_utils.v1 import OpenAIClientV1


class TestOpenAIClientV1Init:
    """Test OpenAIClientV1 initialization."""

    def test_init_with_api_key(self):
        """Test initialization with provided API key."""
        client = OpenAIClientV1(api_key="test-key-123")
        assert client.api_key == "test-key-123"
        assert client.api_base is None

    def test_init_with_api_base(self):
        """Test initialization with custom base URL."""
        client = OpenAIClientV1(
            api_key="test-key-123", api_base="https://custom.api.com"
        )
        assert client.api_key == "test-key-123"
        assert client.api_base == "https://custom.api.com"

    def test_init_with_env_var(self):
        """Test initialization with API key from environment variable."""
        with patch.dict(os.environ, {"OPENAI_API_KEY": "env-key-456"}):
            client = OpenAIClientV1()
            assert client.api_key == "env-key-456"

    @patch("guardrails.utils.openai_utils.v1.openai.Client")
    def test_init_without_api_key(self, _mock_openai_client):
        """Test initialization without API key when env var not set."""
        with patch.dict(os.environ, {}, clear=True):
            client = OpenAIClientV1()
            assert client.api_key is None

    @patch("guardrails.utils.openai_utils.v1.openai.Client")
    def test_client_creation(self, mock_openai_client):
        """Test that OpenAI client is created with correct parameters."""
        OpenAIClientV1(api_key="test-key", api_base="https://custom.api.com")
        mock_openai_client.assert_called_once_with(
            api_key="test-key", base_url="https://custom.api.com"
        )


class TestCreateEmbedding:
    """Test create_embedding method."""

    @patch("guardrails.utils.openai_utils.v1.openai.Client")
    def test_create_embedding_success(self, mock_openai_client):
        """Test successful embedding creation."""
        # Setup mock
        mock_embedding_response = Mock()
        mock_embedding_data_1 = Mock()
        mock_embedding_data_1.embedding = [0.1, 0.2, 0.3]
        mock_embedding_data_2 = Mock()
        mock_embedding_data_2.embedding = [0.4, 0.5, 0.6]
        mock_embedding_response.data = [mock_embedding_data_1, mock_embedding_data_2]

        mock_client_instance = Mock()
        mock_client_instance.embeddings.create.return_value = mock_embedding_response
        mock_openai_client.return_value = mock_client_instance

        # Create client and call method
        client = OpenAIClientV1(api_key="test-key")
        result = client.create_embedding(
            model="text-embedding-ada-002", input=["text1", "text2"]
        )

        # Assertions
        assert result == [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]
        mock_client_instance.embeddings.create.assert_called_once_with(
            model="text-embedding-ada-002", input=["text1", "text2"]
        )

    @patch("guardrails.utils.openai_utils.v1.openai.Client")
    def test_create_embedding_single_input(self, mock_openai_client):
        """Test embedding creation with single input."""
        # Setup mock
        mock_embedding_response = Mock()
        mock_embedding_data = Mock()
        mock_embedding_data.embedding = [0.1, 0.2, 0.3, 0.4, 0.5]
        mock_embedding_response.data = [mock_embedding_data]

        mock_client_instance = Mock()
        mock_client_instance.embeddings.create.return_value = mock_embedding_response
        mock_openai_client.return_value = mock_client_instance

        # Create client and call method
        client = OpenAIClientV1(api_key="test-key")
        result = client.create_embedding(
            model="text-embedding-ada-002", input=["single text"]
        )

        # Assertions
        assert result == [[0.1, 0.2, 0.3, 0.4, 0.5]]
        assert len(result) == 1


class TestCreateCompletion:
    """Test create_completion method."""

    @patch("guardrails.utils.openai_utils.v1.trace_llm_call")
    @patch("guardrails.utils.openai_utils.v1.trace_operation")
    @patch("guardrails.utils.openai_utils.v1.openai.Client")
    def test_create_completion_non_streaming(
        self, mock_openai_client, mock_trace_operation, mock_trace_llm_call
    ):
        """Test non-streaming completion."""
        # Setup mock response
        mock_completion_response = Mock()
        mock_choice = Mock()
        mock_choice.text = "This is a completion response"
        mock_completion_response.choices = [mock_choice]
        mock_usage = Mock()
        mock_usage.prompt_tokens = 10
        mock_usage.completion_tokens = 20
        mock_usage.total_tokens = 30
        mock_completion_response.usage = mock_usage

        mock_client_instance = Mock()
        mock_client_instance.completions.create.return_value = mock_completion_response
        mock_openai_client.return_value = mock_client_instance

        # Create client and call method
        client = OpenAIClientV1(api_key="test-key")
        result = client.create_completion(
            engine="gpt-3.5-turbo-instruct",
            prompt="Test prompt",
            temperature=0.7,
            max_tokens=100,
        )

        # Assertions
        assert isinstance(result, LLMResponse)
        assert result.output == "This is a completion response"
        assert result.prompt_token_count == 10
        assert result.response_token_count == 20

        # Verify API call
        mock_client_instance.completions.create.assert_called_once_with(
            model="gpt-3.5-turbo-instruct",
            prompt="Test prompt",
            temperature=0.7,
            max_tokens=100,
        )

        # Verify tracing calls
        assert mock_trace_operation.call_count == 2  # input and output
        mock_trace_llm_call.assert_called()

    @patch("guardrails.utils.openai_utils.v1.trace_llm_call")
    @patch("guardrails.utils.openai_utils.v1.trace_operation")
    @patch("guardrails.utils.openai_utils.v1.openai.Client")
    def test_create_completion_streaming(
        self, mock_openai_client, mock_trace_operation, mock_trace_llm_call
    ):
        """Test streaming completion."""
        # Setup mock streaming response
        mock_stream = iter([{"choices": [{"text": "chunk1"}]}])

        mock_client_instance = Mock()
        mock_client_instance.completions.create.return_value = mock_stream
        mock_openai_client.return_value = mock_client_instance

        # Create client and call method
        client = OpenAIClientV1(api_key="test-key")
        result = client.create_completion(
            engine="gpt-3.5-turbo-instruct", prompt="Test prompt", stream=True
        )

        # Assertions
        assert isinstance(result, LLMResponse)
        assert result.output == ""
        assert result.stream_output is not None


class TestConstructNonchatResponse:
    """Test construct_nonchat_response method."""

    @patch("guardrails.utils.openai_utils.v1.trace_llm_call")
    def test_construct_nonchat_response_non_streaming(self, mock_trace_llm_call):
        """Test non-streaming response construction."""
        # Setup mock response
        mock_response = Mock()
        mock_choice = Mock()
        mock_choice.text = "Response text"
        mock_response.choices = [mock_choice]
        mock_usage = Mock()
        mock_usage.prompt_tokens = 15
        mock_usage.completion_tokens = 25
        mock_usage.total_tokens = 40
        mock_response.usage = mock_usage

        # Create client and call method
        client = OpenAIClientV1(api_key="test-key")
        result = client.construct_nonchat_response(
            stream=False, openai_response=mock_response
        )

        # Assertions
        assert isinstance(result, LLMResponse)
        assert result.output == "Response text"
        assert result.prompt_token_count == 15
        assert result.response_token_count == 25
        assert result.stream_output is None

    def test_construct_nonchat_response_streaming(self):
        """Test streaming response construction."""
        # Setup mock streaming response
        mock_stream = iter([{"choices": [{"text": "chunk"}]}])

        # Create client and call method
        client = OpenAIClientV1(api_key="test-key")
        result = client.construct_nonchat_response(
            stream=True, openai_response=mock_stream
        )

        # Assertions
        assert isinstance(result, LLMResponse)
        assert result.output == ""
        assert result.stream_output is not None

    def test_construct_nonchat_response_no_choices_error(self):
        """Test error when no choices in response."""
        # Setup mock response with no choices
        mock_response = Mock()
        mock_response.choices = []

        # Create client and call method
        client = OpenAIClientV1(api_key="test-key")

        with pytest.raises(ValueError, match="No choices returned from OpenAI"):
            client.construct_nonchat_response(
                stream=False, openai_response=mock_response
            )

    @patch("guardrails.utils.openai_utils.v1.trace_llm_call")
    def test_construct_nonchat_response_no_usage_error(self, mock_trace_llm_call):
        """Test error when no usage info in response."""
        # Setup mock response with no usage
        mock_response = Mock()
        mock_choice = Mock()
        mock_choice.text = "Response text"
        mock_response.choices = [mock_choice]
        mock_response.usage = None

        # Create client and call method
        client = OpenAIClientV1(api_key="test-key")

        with pytest.raises(ValueError, match="No token counts returned from OpenAI"):
            client.construct_nonchat_response(
                stream=False, openai_response=mock_response
            )


class TestCreateChatCompletion:
    """Test create_chat_completion method."""

    @patch("guardrails.utils.openai_utils.v1.trace_llm_call")
    @patch("guardrails.utils.openai_utils.v1.trace_operation")
    @patch("guardrails.utils.openai_utils.v1.openai.Client")
    def test_create_chat_completion_non_streaming(
        self, mock_openai_client, mock_trace_operation, mock_trace_llm_call
    ):
        """Test non-streaming chat completion."""
        # Setup mock response
        mock_chat_response = Mock()
        mock_choice = Mock()
        mock_message = Mock()
        mock_message.content = "This is a chat response"
        mock_choice.message = mock_message
        mock_chat_response.choices = [mock_choice]
        mock_usage = Mock()
        mock_usage.prompt_tokens = 12
        mock_usage.completion_tokens = 18
        mock_usage.total_tokens = 30
        mock_chat_response.usage = mock_usage

        mock_client_instance = Mock()
        mock_client_instance.chat.completions.create.return_value = mock_chat_response
        mock_openai_client.return_value = mock_client_instance

        # Create client and call method
        messages = [{"role": "user", "content": "Hello"}]
        client = OpenAIClientV1(api_key="test-key")
        result = client.create_chat_completion(
            model="gpt-4", messages=messages, temperature=0.5
        )

        # Assertions
        assert isinstance(result, LLMResponse)
        assert result.output == "This is a chat response"
        assert result.prompt_token_count == 12
        assert result.response_token_count == 18

        # Verify API call
        mock_client_instance.chat.completions.create.assert_called_once_with(
            model="gpt-4", messages=messages, temperature=0.5
        )

    @patch("guardrails.utils.openai_utils.v1.trace_llm_call")
    @patch("guardrails.utils.openai_utils.v1.trace_operation")
    @patch("guardrails.utils.openai_utils.v1.openai.Client")
    def test_create_chat_completion_with_tools(
        self, mock_openai_client, mock_trace_operation, mock_trace_llm_call
    ):
        """Test chat completion with function calling tools."""
        # Setup mock response
        mock_chat_response = Mock()
        mock_choice = Mock()
        mock_message = Mock()
        mock_message.content = "Response with tools"
        mock_choice.message = mock_message
        mock_chat_response.choices = [mock_choice]
        mock_usage = Mock()
        mock_usage.prompt_tokens = 20
        mock_usage.completion_tokens = 30
        mock_usage.total_tokens = 50
        mock_chat_response.usage = mock_usage

        mock_client_instance = Mock()
        mock_client_instance.chat.completions.create.return_value = mock_chat_response
        mock_openai_client.return_value = mock_client_instance

        # Create client and call method
        messages = [{"role": "user", "content": "Call a function"}]
        tools = [
            {
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "description": "Get weather info",
                    "parameters": {},
                },
            }
        ]
        client = OpenAIClientV1(api_key="test-key")
        result = client.create_chat_completion(
            model="gpt-4", messages=messages, tools=tools
        )

        # Assertions
        assert isinstance(result, LLMResponse)
        assert result.output == "Response with tools"

    @patch("guardrails.utils.openai_utils.v1.trace_llm_call")
    @patch("guardrails.utils.openai_utils.v1.trace_operation")
    @patch("guardrails.utils.openai_utils.v1.openai.Client")
    def test_create_chat_completion_streaming(
        self, mock_openai_client, mock_trace_operation, mock_trace_llm_call
    ):
        """Test streaming chat completion."""
        # Setup mock streaming response
        mock_stream = iter([{"choices": [{"delta": {"content": "chunk"}}]}])

        mock_client_instance = Mock()
        mock_client_instance.chat.completions.create.return_value = mock_stream
        mock_openai_client.return_value = mock_client_instance

        # Create client and call method
        messages = [{"role": "user", "content": "Stream this"}]
        client = OpenAIClientV1(api_key="test-key")
        result = client.create_chat_completion(
            model="gpt-4", messages=messages, stream=True
        )

        # Assertions
        assert isinstance(result, LLMResponse)
        assert result.output == ""
        assert result.stream_output is not None


class TestConstructChatResponse:
    """Test construct_chat_response method."""

    @patch("guardrails.utils.openai_utils.v1.trace_llm_call")
    def test_construct_chat_response_with_content(self, mock_trace_llm_call):
        """Test chat response construction with message content."""
        # Setup mock response
        mock_response = Mock()
        mock_choice = Mock()
        mock_message = Mock()
        mock_message.content = "Chat response content"
        mock_choice.message = mock_message
        mock_response.choices = [mock_choice]
        mock_usage = Mock()
        mock_usage.prompt_tokens = 10
        mock_usage.completion_tokens = 15
        mock_usage.total_tokens = 25
        mock_response.usage = mock_usage

        # Create client and call method
        client = OpenAIClientV1(api_key="test-key")
        result = client.construct_chat_response(
            stream=False, openai_response=mock_response
        )

        # Assertions
        assert isinstance(result, LLMResponse)
        assert result.output == "Chat response content"
        assert result.prompt_token_count == 10
        assert result.response_token_count == 15
        assert result.stream_output is None

    @patch("guardrails.utils.openai_utils.v1.trace_llm_call")
    def test_construct_chat_response_with_function_call(self, mock_trace_llm_call):
        """Test chat response construction with function call."""
        # Setup mock response
        mock_response = Mock()
        mock_choice = Mock()
        mock_message = Mock()
        mock_message.content = None
        mock_function_call = Mock()
        mock_function_call.arguments = '{"arg": "value"}'
        mock_message.function_call = mock_function_call
        mock_choice.message = mock_message
        mock_response.choices = [mock_choice]
        mock_usage = Mock()
        mock_usage.prompt_tokens = 20
        mock_usage.completion_tokens = 10
        mock_usage.total_tokens = 30
        mock_response.usage = mock_usage

        # Create client and call method
        client = OpenAIClientV1(api_key="test-key")
        result = client.construct_chat_response(
            stream=False, openai_response=mock_response
        )

        # Assertions
        assert isinstance(result, LLMResponse)
        assert result.output == '{"arg": "value"}'
        assert result.prompt_token_count == 20
        assert result.response_token_count == 10

    @patch("guardrails.utils.openai_utils.v1.trace_llm_call")
    def test_construct_chat_response_with_tool_calls(self, _mock_trace_llm_call):
        """Test chat response construction with tool calls."""

        # Create a custom class for message with controlled attribute access
        class MockMessage:
            content = None

            @property
            def function_call(self):
                raise AttributeError("no function_call")

            @property
            def tool_calls(self):
                mock_tool_call = Mock()
                mock_function = Mock()
                mock_function.arguments = '{"tool_arg": "tool_value"}'
                mock_tool_call.function = mock_function
                return [mock_tool_call]

        # Setup mock response
        mock_response = Mock()
        mock_choice = Mock()
        mock_choice.message = MockMessage()
        mock_response.choices = [mock_choice]
        mock_usage = Mock()
        mock_usage.prompt_tokens = 25
        mock_usage.completion_tokens = 15
        mock_usage.total_tokens = 40
        mock_response.usage = mock_usage

        # Create client and call method
        client = OpenAIClientV1(api_key="test-key")
        result = client.construct_chat_response(
            stream=False, openai_response=mock_response
        )

        # Assertions
        assert isinstance(result, LLMResponse)
        assert result.output == '{"tool_arg": "tool_value"}'
        assert result.prompt_token_count == 25
        assert result.response_token_count == 15

    def test_construct_chat_response_streaming(self):
        """Test streaming chat response construction."""
        # Setup mock streaming response
        mock_stream = iter([{"choices": [{"delta": {"content": "stream chunk"}}]}])

        # Create client and call method
        client = OpenAIClientV1(api_key="test-key")
        result = client.construct_chat_response(
            stream=True, openai_response=mock_stream
        )

        # Assertions
        assert isinstance(result, LLMResponse)
        assert result.output == ""
        assert result.stream_output is not None

    def test_construct_chat_response_no_choices_error(self):
        """Test error when no choices in response."""
        # Setup mock response with no choices
        mock_response = Mock()
        mock_response.choices = []

        # Create client and call method
        client = OpenAIClientV1(api_key="test-key")

        with pytest.raises(ValueError, match="No choices returned from OpenAI"):
            client.construct_chat_response(stream=False, openai_response=mock_response)

    def test_construct_chat_response_no_message_error(self):
        """Test error when no message in response."""
        # Setup mock response with no message
        mock_response = Mock()
        mock_choice = Mock()
        mock_choice.message = None
        mock_response.choices = [mock_choice]

        # Create client and call method
        client = OpenAIClientV1(api_key="test-key")

        with pytest.raises(ValueError, match="No message returned from OpenAI"):
            client.construct_chat_response(stream=False, openai_response=mock_response)

    @patch("guardrails.utils.openai_utils.v1.trace_llm_call")
    def test_construct_chat_response_no_usage_error(self, mock_trace_llm_call):
        """Test error when no usage info in response."""
        # Setup mock response with no usage
        mock_response = Mock()
        mock_choice = Mock()
        mock_message = Mock()
        mock_message.content = "Content"
        mock_choice.message = mock_message
        mock_response.choices = [mock_choice]
        mock_response.usage = None

        # Create client and call method
        client = OpenAIClientV1(api_key="test-key")

        with pytest.raises(ValueError, match="No token counts returned from OpenAI"):
            client.construct_chat_response(stream=False, openai_response=mock_response)

    @patch("guardrails.utils.openai_utils.v1.trace_llm_call")
    def test_construct_chat_response_no_content_or_function_error(
        self, _mock_trace_llm_call
    ):
        """Test error when no content, function_call, or tool_calls in
        response."""

        # Create a custom class for message with controlled attribute access
        class MockMessage:
            content = None

            @property
            def function_call(self):
                raise AttributeError("no function_call")

            @property
            def tool_calls(self):
                raise AttributeError("no tool_calls")

        # Setup mock response with no content or function calls
        mock_response = Mock()
        mock_choice = Mock()
        mock_choice.message = MockMessage()
        mock_response.choices = [mock_choice]
        mock_usage = Mock()
        mock_usage.prompt_tokens = 10
        mock_usage.completion_tokens = 5
        mock_usage.total_tokens = 15
        mock_response.usage = mock_usage

        # Create client and call method
        client = OpenAIClientV1(api_key="test-key")

        with pytest.raises(
            ValueError,
            match="No message content or function call arguments returned from OpenAI",
        ):
            client.construct_chat_response(stream=False, openai_response=mock_response)


class TestIntegrationScenarios:
    """Test integration scenarios combining multiple methods."""

    @patch("guardrails.utils.openai_utils.v1.trace_llm_call")
    @patch("guardrails.utils.openai_utils.v1.trace_operation")
    @patch("guardrails.utils.openai_utils.v1.openai.Client")
    def test_full_completion_workflow(
        self, mock_openai_client, mock_trace_operation, mock_trace_llm_call
    ):
        """Test complete workflow from client creation to completion."""
        # Setup mock
        mock_completion_response = Mock()
        mock_choice = Mock()
        mock_choice.text = "Full workflow response"
        mock_completion_response.choices = [mock_choice]
        mock_usage = Mock()
        mock_usage.prompt_tokens = 5
        mock_usage.completion_tokens = 10
        mock_usage.total_tokens = 15
        mock_completion_response.usage = mock_usage

        mock_client_instance = Mock()
        mock_client_instance.completions.create.return_value = mock_completion_response
        mock_openai_client.return_value = mock_client_instance

        # Execute workflow
        client = OpenAIClientV1(api_key="workflow-test-key")
        result = client.create_completion(
            engine="gpt-3.5-turbo-instruct", prompt="Test workflow"
        )

        # Verify end-to-end result
        assert result.output == "Full workflow response"
        assert result.prompt_token_count == 5
        assert result.response_token_count == 10

    @patch("guardrails.utils.openai_utils.v1.trace_llm_call")
    @patch("guardrails.utils.openai_utils.v1.trace_operation")
    @patch("guardrails.utils.openai_utils.v1.openai.Client")
    def test_full_chat_workflow(
        self, mock_openai_client, mock_trace_operation, mock_trace_llm_call
    ):
        """Test complete workflow for chat completion."""
        # Setup mock
        mock_chat_response = Mock()
        mock_choice = Mock()
        mock_message = Mock()
        mock_message.content = "Full chat workflow response"
        mock_choice.message = mock_message
        mock_chat_response.choices = [mock_choice]
        mock_usage = Mock()
        mock_usage.prompt_tokens = 8
        mock_usage.completion_tokens = 12
        mock_usage.total_tokens = 20
        mock_chat_response.usage = mock_usage

        mock_client_instance = Mock()
        mock_client_instance.chat.completions.create.return_value = mock_chat_response
        mock_openai_client.return_value = mock_client_instance

        # Execute workflow
        client = OpenAIClientV1(api_key="chat-workflow-key")
        messages = [
            {"role": "system", "content": "You are a helpful assistant"},
            {"role": "user", "content": "Hello"},
        ]
        result = client.create_chat_completion(model="gpt-4", messages=messages)

        # Verify end-to-end result
        assert result.output == "Full chat workflow response"
        assert result.prompt_token_count == 8
        assert result.response_token_count == 12
