from .test_assets import entity_extraction, pydantic, python_rail, string


def openai_completion_create(prompt, *args, **kwargs):
    """Mock the OpenAI API call to Completion.create."""
    # NOTE: this function normally overrides `llm_providers.openai_wrapper`,
    # which compiles instructions and prompt into a single prompt
    # here the instructions are passed into kwargs and ignored

    mock_llm_responses = {
        entity_extraction.COMPILED_PROMPT: entity_extraction.LLM_OUTPUT,
        entity_extraction.COMPILED_PROMPT_REASK: entity_extraction.LLM_OUTPUT_REASK,
        entity_extraction.COMPILED_PROMPT_FULL_REASK: entity_extraction.LLM_OUTPUT_FULL_REASK,  # noqa: E501
        entity_extraction.COMPILED_PROMPT_SKELETON_REASK_1: entity_extraction.LLM_OUTPUT_SKELETON_REASK_1,  # noqa: E501
        entity_extraction.COMPILED_PROMPT_SKELETON_REASK_2: entity_extraction.LLM_OUTPUT_SKELETON_REASK_2,  # noqa: E501
        pydantic.COMPILED_PROMPT: pydantic.LLM_OUTPUT,
        pydantic.COMPILED_PROMPT_REASK_1: pydantic.LLM_OUTPUT_REASK_1,
        pydantic.COMPILED_PROMPT_REASK_2: pydantic.LLM_OUTPUT_REASK_2,
        string.COMPILED_PROMPT: string.LLM_OUTPUT,
        string.COMPILED_PROMPT_REASK: string.LLM_OUTPUT_REASK,
        python_rail.VALIDATOR_PARALLELISM_PROMPT_1: python_rail.VALIDATOR_PARALLELISM_RESPONSE_1,  # noqa: E501
        python_rail.VALIDATOR_PARALLELISM_PROMPT_2: python_rail.VALIDATOR_PARALLELISM_RESPONSE_2,  # noqa: E501
        python_rail.VALIDATOR_PARALLELISM_PROMPT_3: python_rail.VALIDATOR_PARALLELISM_RESPONSE_3,  # noqa: E501
    }

    try:
        return mock_llm_responses[prompt]
    except KeyError:
        print(prompt)
        raise ValueError("Compiled prompt not found")


async def async_openai_completion_create(prompt, *args, **kwargs):
    return openai_completion_create(prompt, *args, **kwargs)


def openai_chat_completion_create(
    prompt=None, instructions=None, msg_history=None, base_model=None, *args, **kwargs
):
    """Mock the OpenAI API call to ChatCompletion.create."""

    mock_llm_responses = {
        (
            entity_extraction.COMPILED_PROMPT_WITHOUT_INSTRUCTIONS,
            entity_extraction.COMPILED_INSTRUCTIONS,
        ): entity_extraction.LLM_OUTPUT,
        (
            entity_extraction.COMPILED_PROMPT_REASK,
            entity_extraction.COMPILED_INSTRUCTIONS_REASK,
        ): entity_extraction.LLM_OUTPUT_REASK,
        (
            python_rail.COMPILED_PROMPT_1_WITHOUT_INSTRUCTIONS,
            python_rail.COMPILED_INSTRUCTIONS,
        ): python_rail.LLM_OUTPUT_1_FAIL_GUARDRAILS_VALIDATION,
        (
            python_rail.COMPILED_PROMPT_2_WITHOUT_INSTRUCTIONS,
            python_rail.COMPILED_INSTRUCTIONS,
        ): python_rail.LLM_OUTPUT_2_SUCCEED_GUARDRAILS_BUT_FAIL_PYDANTIC_VALIDATION,
        (
            string.MSG_COMPILED_PROMPT_REASK,
            string.MSG_COMPILED_INSTRUCTIONS_REASK,
        ): string.MSG_LLM_OUTPUT_CORRECT,
        (
            pydantic.MSG_COMPILED_PROMPT_REASK,
            pydantic.MSG_COMPILED_INSTRUCTIONS_REASK,
        ): pydantic.MSG_HISTORY_LLM_OUTPUT_CORRECT,
    }
    try:
        if prompt and instructions and not msg_history:
            return mock_llm_responses[(prompt, instructions)]
        elif msg_history and not prompt and not instructions:
            if msg_history == entity_extraction.COMPILED_MSG_HISTORY:
                return entity_extraction.LLM_OUTPUT
            elif (
                msg_history == string.MOVIE_MSG_HISTORY
                and base_model == pydantic.WITH_MSG_HISTORY
            ):
                return pydantic.MSG_HISTORY_LLM_OUTPUT_INCORRECT
            elif msg_history == string.MOVIE_MSG_HISTORY:
                return string.MSG_LLM_OUTPUT_INCORRECT
            else:
                raise ValueError("msg_history not found")
    except KeyError:
        raise ValueError("Compiled prompt not found")
