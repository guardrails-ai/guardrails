from guardrails.llm_providers import (
    ArbitraryCallable,
    AsyncArbitraryCallable,
    AsyncLiteLLMCallable,
    LiteLLMCallable,
)
from guardrails.classes.llm.llm_response import LLMResponse

from .test_assets import entity_extraction, pydantic, python_rail, string


class MockAsyncLiteLLMCallable(AsyncLiteLLMCallable):
    async def invoke_llm(self, prompt, *args, **kwargs):
        sync_mock = MockLiteLLMCallable()
        return sync_mock._invoke_llm(prompt, *args, **kwargs)


class MockLiteLLMCallable(LiteLLMCallable):
    def _invoke_llm(
        self,
        prompt=None,
        instructions=None,
        msg_history=None,
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
            if prompt and instructions and not msg_history:
                out_text = mock_llm_responses[(prompt, instructions)]
            elif msg_history and not prompt and not instructions:
                if msg_history == entity_extraction.COMPILED_MSG_HISTORY:
                    out_text = entity_extraction.LLM_OUTPUT
                elif (
                    msg_history == string.MOVIE_MSG_HISTORY
                    and base_model == pydantic.WITH_MSG_HISTORY
                ):
                    out_text = pydantic.MSG_HISTORY_LLM_OUTPUT_INCORRECT
                elif msg_history == string.MOVIE_MSG_HISTORY:
                    out_text = string.MSG_LLM_OUTPUT_INCORRECT
                else:
                    raise ValueError("msg_history not found")
            else:
                raise ValueError(
                    "specify either prompt and instructions " "or msg_history"
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
            print("\n msg_history: \n", msg_history)
            raise ValueError("Compiled prompt not found")


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
