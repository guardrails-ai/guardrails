from .test_cases import entity_extraction


def openai_completion_create(prompt, *args, **kwargs):
    """Mock the OpenAI API call to Completion.create."""

    mock_llm_responses = {
        entity_extraction.COMPILED_PROMPT: entity_extraction.LLM_OUTPUT,
        entity_extraction.COMPILED_PROMPT_REASK: entity_extraction.LLM_OUTPUT_REASK,
    }

    try:
        return mock_llm_responses[prompt]
    except KeyError:
        raise ValueError("Compiled prompt not found")
