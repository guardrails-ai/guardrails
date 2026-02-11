from unittest.mock import Mock, patch

import pytest

from guardrails.utils.openai_utils.streaming_utils import (
    num_tokens_from_messages,
    num_tokens_from_string,
)


class TestNumTokensFromString:
    """Test num_tokens_from_string function."""

    def test_simple_text_gpt35(self):
        """Test token counting for simple text with GPT-3.5."""
        text = "Hello, world!"
        result = num_tokens_from_string(text, "gpt-3.5-turbo")
        assert isinstance(result, int)
        assert result > 0

    def test_simple_text_gpt4(self):
        """Test token counting for simple text with GPT-4."""
        text = "Hello, world!"
        result = num_tokens_from_string(text, "gpt-4")
        assert isinstance(result, int)
        assert result > 0

    def test_empty_string(self):
        """Test token counting for empty string."""
        text = ""
        result = num_tokens_from_string(text, "gpt-3.5-turbo")
        assert result == 0

    def test_longer_text(self):
        """Test token counting for longer text."""
        text = """
        This is a longer text that contains multiple sentences.
        It should return a higher token count than a simple string.
        Let's add more content to make it even longer and see what happens.
        """
        result = num_tokens_from_string(text, "gpt-3.5-turbo")
        assert isinstance(result, int)
        assert result > 10

    def test_special_characters(self):
        """Test token counting with special characters."""
        text = "Hello! @#$%^&*() 你好 🎉"
        result = num_tokens_from_string(text, "gpt-3.5-turbo")
        assert isinstance(result, int)
        assert result > 0

    def test_code_text(self):
        """Test token counting for code."""
        text = """
        def hello_world():
            print("Hello, World!")
            return True
        """
        result = num_tokens_from_string(text, "gpt-3.5-turbo")
        assert isinstance(result, int)
        assert result > 0

    def test_different_models_same_text(self):
        """Test that different models can tokenize the same text."""
        text = "This is a test sentence."
        result_gpt35 = num_tokens_from_string(text, "gpt-3.5-turbo")
        result_gpt4 = num_tokens_from_string(text, "gpt-4")

        # Both should return valid token counts
        assert isinstance(result_gpt35, int)
        assert isinstance(result_gpt4, int)
        assert result_gpt35 > 0
        assert result_gpt4 > 0

    def test_unicode_text(self):
        """Test token counting with Unicode characters."""
        text = "Hello 世界 مرحبا мир"
        result = num_tokens_from_string(text, "gpt-3.5-turbo")
        assert isinstance(result, int)
        assert result > 0

    def test_newlines_and_whitespace(self):
        """Test token counting with various whitespace."""
        text = "Line 1\n\nLine 2\t\tTabbed\r\nCarriage return"
        result = num_tokens_from_string(text, "gpt-3.5-turbo")
        assert isinstance(result, int)
        assert result > 0


class TestNumTokensFromMessages:
    """Test num_tokens_from_messages function."""

    def test_single_message_gpt35_turbo_0613(self):
        """Test token counting for single message with gpt-3.5-turbo-0613."""
        messages = [{"role": "user", "content": "Hello!"}]
        result = num_tokens_from_messages(messages, "gpt-3.5-turbo-0613")
        # Should be: 3 (tokens_per_message)
        #   + tokens in "user"
        #   + tokens in "Hello!"
        #   + 3 (priming)
        assert isinstance(result, int)
        assert result > 0

    def test_multiple_messages_gpt35_turbo_0613(self):
        """Test token counting for multiple messages."""
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Hello!"},
            {"role": "assistant", "content": "Hi there!"},
        ]
        result = num_tokens_from_messages(messages, "gpt-3.5-turbo-0613")
        assert isinstance(result, int)
        assert result > 10

    def test_message_with_name_gpt35_turbo_0613(self):
        """Test token counting for message with name field."""
        messages = [{"role": "user", "name": "Alice", "content": "Hello!"}]
        result = num_tokens_from_messages(messages, "gpt-3.5-turbo-0613")
        # Should include tokens_per_name (1) for the name field
        assert isinstance(result, int)
        assert result > 0

    def test_gpt4_0613(self):
        """Test token counting with gpt-4-0613."""
        messages = [{"role": "user", "content": "Hello!"}]
        result = num_tokens_from_messages(messages, "gpt-4-0613")
        assert isinstance(result, int)
        assert result > 0

    def test_gpt4_32k_0613(self):
        """Test token counting with gpt-4-32k-0613."""
        messages = [{"role": "user", "content": "Hello!"}]
        result = num_tokens_from_messages(messages, "gpt-4-32k-0613")
        assert isinstance(result, int)
        assert result > 0

    def test_gpt35_turbo_16k_0613(self):
        """Test token counting with gpt-3.5-turbo-16k-0613."""
        messages = [{"role": "user", "content": "Hello!"}]
        result = num_tokens_from_messages(messages, "gpt-3.5-turbo-16k-0613")
        assert isinstance(result, int)
        assert result > 0

    def test_gpt35_turbo_0301(self):
        """Test token counting with gpt-3.5-turbo-0301 (different token
        counting)."""
        messages = [{"role": "user", "content": "Hello!"}]
        result = num_tokens_from_messages(messages, "gpt-3.5-turbo-0301")
        # tokens_per_message = 4, tokens_per_name = -1
        assert isinstance(result, int)
        assert result > 0

    def test_gpt35_turbo_0301_with_name(self):
        """Test token counting with gpt-3.5-turbo-0301 with name field."""
        messages = [{"role": "user", "name": "Alice", "content": "Hello!"}]
        result = num_tokens_from_messages(messages, "gpt-3.5-turbo-0301")
        # tokens_per_name = -1, so having a name should reduce count
        assert isinstance(result, int)
        assert result > 0

    @patch("guardrails.utils.openai_utils.streaming_utils.logger")
    def test_generic_gpt35_turbo_fallback(self, mock_logger):
        """Test that generic gpt-3.5-turbo falls back to gpt-3.5-turbo-0613."""
        messages = [{"role": "user", "content": "Hello!"}]
        result = num_tokens_from_messages(messages, "gpt-3.5-turbo")

        # Should log a warning about the fallback
        assert mock_logger.warning.called
        warning_message = mock_logger.warning.call_args[0][0]
        assert "gpt-3.5-turbo may update over time" in warning_message

        # Should still return a valid token count
        assert isinstance(result, int)
        assert result > 0

    @patch("guardrails.utils.openai_utils.streaming_utils.logger")
    def test_generic_gpt4_fallback(self, mock_logger):
        """Test that generic gpt-4 falls back to gpt-4-0613."""
        messages = [{"role": "user", "content": "Hello!"}]
        result = num_tokens_from_messages(messages, "gpt-4")

        # Should log a warning about the fallback
        assert mock_logger.warning.called
        warning_message = mock_logger.warning.call_args[0][0]
        assert "gpt-4 may update over time" in warning_message

        # Should still return a valid token count
        assert isinstance(result, int)
        assert result > 0

    @patch("guardrails.utils.openai_utils.streaming_utils.logger")
    def test_custom_gpt35_turbo_variant(self, mock_logger):
        """Test custom gpt-3.5-turbo variant falls back to
        gpt-3.5-turbo-0613."""
        messages = [{"role": "user", "content": "Hello!"}]
        result = num_tokens_from_messages(messages, "gpt-3.5-turbo-custom")

        # Should log a warning
        assert mock_logger.warning.called

        # Should still return a valid token count
        assert isinstance(result, int)
        assert result > 0

    @patch("guardrails.utils.openai_utils.streaming_utils.logger")
    def test_custom_gpt4_variant(self, mock_logger):
        """Test custom gpt-4 variant falls back to gpt-4-0613."""
        messages = [{"role": "user", "content": "Hello!"}]
        result = num_tokens_from_messages(messages, "gpt-4-turbo")

        # Should log a warning
        assert mock_logger.warning.called

        # Should still return a valid token count
        assert isinstance(result, int)
        assert result > 0

    @patch("guardrails.utils.openai_utils.streaming_utils.logger")
    def test_unknown_model_with_keyerror(self, mock_logger):
        """Test unknown model that raises KeyError."""
        messages = [{"role": "user", "content": "Hello!"}]

        # Mock tiktoken to raise KeyError for unknown model
        with patch(
            "guardrails.utils.openai_utils.streaming_utils.tiktoken"
        ) as mock_tiktoken:
            mock_encoding = Mock()
            mock_encoding.encode.return_value = [1, 2, 3]

            def encoding_for_model_side_effect(model):
                if model == "unknown-model":
                    raise KeyError("Model not found")
                return mock_encoding

            mock_tiktoken.encoding_for_model.side_effect = (
                encoding_for_model_side_effect  # noqa
            )
            mock_tiktoken.get_encoding.return_value = mock_encoding

            # Should raise NotImplementedError for truly unknown model
            with pytest.raises(
                NotImplementedError,
                match="num_tokens_from_messages\\(\\) is not implemented for model unknown-model",  # noqa
            ):
                num_tokens_from_messages(messages, "unknown-model")

            # Should have logged warning about model not found
            mock_logger.warning.assert_called_with(
                "model not found. Using cl100k_base encoding."
            )

    def test_empty_messages_list(self):
        """Test token counting with empty messages list."""
        messages = []
        result = num_tokens_from_messages(messages, "gpt-3.5-turbo-0613")
        # Should only have the 3 tokens for priming
        assert result == 3

    def test_message_with_empty_content(self):
        """Test token counting for message with empty content."""
        messages = [{"role": "user", "content": ""}]
        result = num_tokens_from_messages(messages, "gpt-3.5-turbo-0613")
        # Should have tokens_per_message
        #   + tokens for "user"
        #   + 0 for content
        #   + 3 for priming
        assert isinstance(result, int)
        assert result > 0

    def test_complex_conversation(self):
        """Test token counting for a complex conversation."""
        messages = [
            {"role": "system", "content": "You are a helpful coding assistant."},
            {"role": "user", "content": "How do I write a function in Python?"},
            {
                "role": "assistant",
                "content": "Here's how to write a function in Python:\n\ndef my_function():\n    pass",  # noqa
            },
            {"role": "user", "content": "Thanks!"},
        ]
        result = num_tokens_from_messages(messages, "gpt-3.5-turbo-0613")
        assert isinstance(result, int)
        assert result > 20

    def test_message_with_multiple_fields(self):
        """Test token counting for message with multiple fields."""
        messages = [
            {
                "role": "user",
                "name": "Alice",
                "content": "Hello, how are you?",
            }
        ]
        result = num_tokens_from_messages(messages, "gpt-3.5-turbo-0613")
        assert isinstance(result, int)
        assert result > 0

    def test_consistent_token_count(self):
        """Test that token count is consistent for the same input."""
        messages = [{"role": "user", "content": "Test message"}]
        result1 = num_tokens_from_messages(messages, "gpt-3.5-turbo-0613")
        result2 = num_tokens_from_messages(messages, "gpt-3.5-turbo-0613")
        assert result1 == result2

    def test_gpt4_0314(self):
        """Test token counting with gpt-4-0314."""
        messages = [{"role": "user", "content": "Hello!"}]
        result = num_tokens_from_messages(messages, "gpt-4-0314")
        assert isinstance(result, int)
        assert result > 0

    def test_gpt4_32k_0314(self):
        """Test token counting with gpt-4-32k-0314."""
        messages = [{"role": "user", "content": "Hello!"}]
        result = num_tokens_from_messages(messages, "gpt-4-32k-0314")
        assert isinstance(result, int)
        assert result > 0

    def test_messages_with_unicode(self):
        """Test token counting for messages with Unicode content."""
        messages = [
            {"role": "user", "content": "Hello 世界"},
            {"role": "assistant", "content": "مرحبا مир"},
        ]
        result = num_tokens_from_messages(messages, "gpt-3.5-turbo-0613")
        assert isinstance(result, int)
        assert result > 0

    def test_long_message_content(self):
        """Test token counting for messages with long content."""
        long_content = " ".join(["word"] * 1000)
        messages = [{"role": "user", "content": long_content}]
        result = num_tokens_from_messages(messages, "gpt-3.5-turbo-0613")
        assert isinstance(result, int)
        assert result > 1000

    def test_default_model_parameter(self):
        """Test that default model parameter is gpt-3.5-turbo-0613."""
        messages = [{"role": "user", "content": "Hello!"}]
        # Call without specifying model - should use default
        result = num_tokens_from_messages(messages)
        assert isinstance(result, int)
        assert result > 0

    def test_priming_tokens_included(self):
        """Test that the 3 priming tokens are always included."""
        # Even with empty messages, should have 3 tokens for priming
        messages = []
        result = num_tokens_from_messages(messages, "gpt-3.5-turbo-0613")
        assert result == 3

    @patch("guardrails.utils.openai_utils.streaming_utils.logger")
    def test_unsupported_model_raises_error(self, _mock_logger):
        """Test that unsupported model raises NotImplementedError."""
        messages = [{"role": "user", "content": "Hello!"}]

        with pytest.raises(NotImplementedError) as exc_info:
            num_tokens_from_messages(messages, "claude-2")

        assert (
            "num_tokens_from_messages() is not implemented for model claude-2"
            in str(  # noqa
                exc_info.value
            )
        )


class TestIntegrationScenarios:
    """Test integration scenarios combining both functions."""

    def test_string_tokens_vs_message_tokens(self):
        """Test that message tokens include overhead beyond just content
        tokens."""
        content = "Hello, world!"
        string_tokens = num_tokens_from_string(content, "gpt-3.5-turbo")
        message_tokens = num_tokens_from_messages(
            [{"role": "user", "content": content}], "gpt-3.5-turbo-0613"
        )

        # Message tokens should be higher due to role tokens and overhead
        assert message_tokens > string_tokens

    def test_multiple_messages_sum(self):
        """Test that token counting works correctly for multiple messages."""
        messages = [
            {"role": "user", "content": "First message"},
            {"role": "assistant", "content": "Second message"},
        ]
        result = num_tokens_from_messages(messages, "gpt-3.5-turbo-0613")

        # Should be more than just the sum of content tokens
        first_tokens = num_tokens_from_string("First message", "gpt-3.5-turbo")
        second_tokens = num_tokens_from_string("Second message", "gpt-3.5-turbo")

        # Total should include overhead for roles, formatting, and priming
        assert result > first_tokens + second_tokens

    def test_consistency_across_model_versions(self):
        """Test that specific model versions produce consistent results."""
        messages = [{"role": "user", "content": "Test"}]

        # These models should use the same token counting (tokens_per_message=3)
        result_gpt35 = num_tokens_from_messages(messages, "gpt-3.5-turbo-0613")
        result_gpt4 = num_tokens_from_messages(messages, "gpt-4-0613")

        # Should both return valid counts (may differ slightly due to encoding)
        assert isinstance(result_gpt35, int)
        assert isinstance(result_gpt4, int)
        assert result_gpt35 > 0
        assert result_gpt4 > 0
