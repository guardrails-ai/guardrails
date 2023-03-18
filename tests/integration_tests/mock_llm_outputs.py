from .test_cases import entity_extraction


def openai_Completion_create(prompt, *args, **kwargs):
    """Mock the OpenAI API call to Completion.create."""
    if prompt == entity_extraction.COMPILED_PROMPT_REASK_1:
        return entity_extraction.LLM_OUTPUT_REASK_1
    elif prompt == entity_extraction.COMPILED_PROMPT_REASK_2:
        return entity_extraction.LLM_OUTPUT_REASK_2
    else:
        raise ValueError("Compiled prompt not found")
