from .test_cases import entity_extraction, pydantic


def openai_completion_create(prompt, *args, **kwargs):
    """Mock the OpenAI API call to Completion.create."""
    mock_llm_responses = {
        entity_extraction.COMPILED_PROMPT: entity_extraction.LLM_OUTPUT,
        entity_extraction.COMPILED_PROMPT_REASK: entity_extraction.LLM_OUTPUT_REASK,
        pydantic.COMPILED_PROMPT: pydantic.LLM_OUTPUT,
        pydantic.COMPILED_PROMPT_REASK_1: pydantic.LLM_OUTPUT_REASK_1,
        pydantic.COMPILED_PROMPT_REASK_2: pydantic.LLM_OUTPUT_REASK_2,
    }

    try:
        return mock_llm_responses[prompt]
    except KeyError:
        raise ValueError("Compiled prompt not found")


def openai_chat_completion_create(prompt, instructions, *args, **kwargs):
    """Mock the OpenAI API call to ChatCompletion.create."""
    mock_llm_responses = {
        (
            entity_extraction.COMPILED_PROMPT_WITHOUT_INSTRUCTIONS,
            entity_extraction.COMPILED_INSTRUCTIONS,
        ): entity_extraction.LLM_OUTPUT,
    }

    try:
        return mock_llm_responses[(prompt, instructions)]
    except KeyError:
        raise ValueError("Compiled prompt not found")
