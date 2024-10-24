from guardrails.llm_providers import (
    ArbitraryCallable,
    AsyncArbitraryCallable,
    AsyncLiteLLMCallable,
    LiteLLMCallable,
)
from guardrails.classes.llm.llm_response import LLMResponse

from .test_assets import entity_extraction, lists_object, pydantic, python_rail, string


class MockLiteLLMCallableOther(LiteLLMCallable):
    # NOTE: this class normally overrides `llm_providers.LiteLLMCallable`,
    # which compiles instructions and prompt into a single prompt;
    # here the instructions are passed into kwargs and ignored
    def _invoke_llm(self, messages, *args, **kwargs):
        """Mock the OpenAI API call to Completion.create."""

        _rail_to_compiled_prompt = {  # noqa
            entity_extraction.RAIL_SPEC_WITH_REASK: entity_extraction.COMPILED_PROMPT,
        }

        mock_llm_responses = {
            entity_extraction.COMPILED_PROMPT: entity_extraction.LLM_OUTPUT,
            entity_extraction.COMPILED_PROMPT_REASK: entity_extraction.LLM_OUTPUT_REASK,
            entity_extraction.COMPILED_PROMPT_FULL_REASK: entity_extraction.LLM_OUTPUT_FULL_REASK,  # noqa: E501
            entity_extraction.COMPILED_PROMPT_SKELETON_REASK_1: entity_extraction.LLM_OUTPUT_SKELETON_REASK_1,  # noqa: E501
            entity_extraction.COMPILED_PROMPT_SKELETON_REASK_2: entity_extraction.LLM_OUTPUT_SKELETON_REASK_2,  # noqa: E501
            pydantic.COMPILED_PROMPT: pydantic.LLM_OUTPUT,
            pydantic.COMPILED_PROMPT_REASK_1: pydantic.LLM_OUTPUT_REASK_1,
            pydantic.COMPILED_PROMPT_FULL_REASK_1: pydantic.LLM_OUTPUT_FULL_REASK_1,
            pydantic.COMPILED_PROMPT_REASK_2: pydantic.LLM_OUTPUT_REASK_2,
            pydantic.COMPILED_PROMPT_FULL_REASK_2: pydantic.LLM_OUTPUT_FULL_REASK_2,
            pydantic.COMPILED_PROMPT_ENUM: pydantic.LLM_OUTPUT_ENUM,
            pydantic.COMPILED_PROMPT_ENUM_2: pydantic.LLM_OUTPUT_ENUM_2,
            string.COMPILED_PROMPT: string.LLM_OUTPUT,
            string.COMPILED_PROMPT_REASK: string.LLM_OUTPUT_REASK,
            string.COMPILED_LIST_PROMPT: string.LIST_LLM_OUTPUT,
            python_rail.VALIDATOR_PARALLELISM_PROMPT_1: python_rail.VALIDATOR_PARALLELISM_RESPONSE_1,  # noqa: E501
            python_rail.VALIDATOR_PARALLELISM_PROMPT_2: python_rail.VALIDATOR_PARALLELISM_RESPONSE_2,  # noqa: E501
            python_rail.VALIDATOR_PARALLELISM_PROMPT_3: python_rail.VALIDATOR_PARALLELISM_RESPONSE_3,  # noqa: E501
            lists_object.LIST_PROMPT: lists_object.LIST_OUTPUT,
        }

        try:
            output = mock_llm_responses[messages[0]["content"]]
            return LLMResponse(
                output=output,
                prompt_token_count=123,
                response_token_count=1234,
            )
        except KeyError:
            print("Unrecognized messages!")
            print(messages)
            raise ValueError("Compiled messages not found")


class MockAsyncLiteLLMCallable(AsyncLiteLLMCallable):
    async def invoke_llm(self, prompt, *args, **kwargs):
        sync_mock = MockLiteLLMCallable()
        return sync_mock._invoke_llm(prompt, *args, **kwargs)


class MockLiteLLMCallable(LiteLLMCallable):
    def _invoke_llm(
        self,
        prompt=None,
        instructions=None,
        messages=None,
        base_model=None,
        *args,
        **kwargs,
    ):
        """Mock the OpenAI API call to ChatCompletion.create."""

        _rail_to_prompt = {
            entity_extraction.RAIL_SPEC_WITH_FIX_CHAT_MODEL: (
                entity_extraction.COMPILED_PROMPT_WITHOUT_INSTRUCTIONS,
                entity_extraction.COMPILED_INSTRUCTIONS,
            )
        }

        mock_llm_responses = {
            (
                entity_extraction.COMPILED_PROMPT_WITHOUT_INSTRUCTIONS,
                entity_extraction.COMPILED_INSTRUCTIONS,
            ): entity_extraction.LLM_OUTPUT,
            (
                entity_extraction.COMPILED_PROMPT_REASK_WITHOUT_INSTRUCTIONS,
                entity_extraction.COMPILED_INSTRUCTIONS_REASK,
            ): entity_extraction.LLM_OUTPUT_REASK,
            (
                python_rail.COMPILED_PROMPT_1_WITHOUT_INSTRUCTIONS,
                python_rail.COMPILED_INSTRUCTIONS,
            ): python_rail.LLM_OUTPUT_1_FAIL_GUARDRAILS_VALIDATION,
            (
                python_rail.COMPILED_PROMPT_1_PYDANTIC_2_WITHOUT_INSTRUCTIONS,
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
            (
                pydantic.COMPILED_PROMPT_CHAT,
                pydantic.COMPILED_INSTRUCTIONS_CHAT,
            ): pydantic.LLM_OUTPUT,
            (
                pydantic.COMPILED_PROMPT_FULL_REASK_1,
                pydantic.COMPILED_INSTRUCTIONS_CHAT,
            ): pydantic.LLM_OUTPUT_FULL_REASK_1,
            (
                pydantic.COMPILED_PROMPT_FULL_REASK_2,
                pydantic.COMPILED_INSTRUCTIONS_CHAT,
            ): pydantic.LLM_OUTPUT_FULL_REASK_2,
            (
                string.PARSE_COMPILED_PROMPT_REASK,
                string.MSG_COMPILED_INSTRUCTIONS_REASK,
            ): string.MSG_LLM_OUTPUT_CORRECT,
        }

        try:
            out_text = None
            if messages:
                if len(messages) == 2:
                    key = (messages[0]["content"], messages[1]["content"])
                elif len(messages) == 1:
                    key = (messages[0]["content"], None)

                if hasattr(mock_llm_responses[key], "read"):
                    out_text = mock_llm_responses[key]
            else:
                raise ValueError(
                    "specify either prompt and instructions " "or messages"
                )
            return LLMResponse(
                output=out_text,
                prompt_token_count=123,
                response_token_count=1234,
            )
        except KeyError:
            print("Unrecognized prompt!")
            print("\n prompt: \n", prompt)
            print("\n instructions: \n", instructions)
            print("\n messages: \n", messages)
            print("\n base_model: \n", base_model)
            raise ValueError("Compiled prompt not found in mock llm response")


class MockArbitraryCallable(ArbitraryCallable):
    # NOTE: this class normally overrides `llm_providers.ArbitraryCallable`,
    # which compiles instructions and prompt into a single prompt;
    # here the instructions are passed into kwargs and ignored
    def _invoke_llm(self, prompt, *args, **kwargs):
        """Mock an arbitrary callable."""

        mock_llm_responses = {
            pydantic.PARSING_COMPILED_PROMPT: pydantic.PARSING_UNPARSEABLE_LLM_OUTPUT,
            pydantic.PARSING_COMPILED_REASK: pydantic.PARSING_EXPECTED_LLM_OUTPUT,
        }

        try:
            return LLMResponse(
                output=mock_llm_responses[prompt],
                prompt_token_count=123,
                response_token_count=1234,
            )
        except KeyError:
            print(prompt)
            raise ValueError("Compiled prompt not found")


class MockAsyncArbitraryCallable(AsyncArbitraryCallable):
    async def invoke_llm(self, prompt, *args, **kwargs):
        sync_mock = MockArbitraryCallable(kwargs.get("llm_api"))
        return sync_mock._invoke_llm(prompt, *args, **kwargs)
