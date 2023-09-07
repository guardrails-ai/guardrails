def mock_chat_completion(**kwargs):
    """Mocks the OpenAI chat completion function for ProvenanceV1."""
    return {"choices": [{"message": {"content": "Yes"}}]}


def mock_chromadb_query_function(**kwargs):
    """Mocks the ChromaDB query function for ProvenanceV1."""
    return [
        "Lorem ipsum dolor sit amet, consectetur adipiscing elit.",
        "Sed do eiusmod tempor incididunt ut labore et dolore magna aliqua.",
        "Ut enim ad minim veniam.",
    ]
