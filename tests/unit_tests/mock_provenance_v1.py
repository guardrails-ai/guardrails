from guardrails.utils.openai_utils import OPENAI_VERSION


def mock_chat_completion(*args, **kwargs):
    """Mocks the OpenAI chat completion function for ProvenanceV1."""
    if OPENAI_VERSION.startswith("0"):
        return {
            "choices": [{"message": {"content": "Yes"}}],
            "usage": {
                "prompt_tokens": 10,
                "completion_tokens": 20,
            },
        }
    else:
        from openai.types import CompletionUsage
        from openai.types.chat import ChatCompletion, ChatCompletionMessage
        from openai.types.chat.chat_completion import Choice

        return ChatCompletion(
            id="",
            choices=[
                Choice(
                    finish_reason="stop",
                    index=0,
                    message=ChatCompletionMessage(
                        content="Yes",
                        role="assistant",
                    ),
                )
            ],
            created=0,
            model="",
            object="chat.completion",
            usage=CompletionUsage(
                prompt_tokens=10,
                completion_tokens=20,
                total_tokens=30,
            ),
        )


def mock_chromadb_query_function(**kwargs):
    """Mocks the ChromaDB query function for ProvenanceV1."""
    return [
        "Lorem ipsum dolor sit amet, consectetur adipiscing elit.",
        "Sed do eiusmod tempor incididunt ut labore et dolore magna aliqua.",
        "Ut enim ad minim veniam.",
    ]
