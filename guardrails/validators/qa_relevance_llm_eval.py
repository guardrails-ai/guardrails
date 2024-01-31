import inspect
from typing import Any, Callable, Dict, Optional, cast

from guardrails.utils.openai_utils import get_static_openai_chat_create_func
from guardrails.validator_base import (
    FailResult,
    PassResult,
    ValidationResult,
    Validator,
    register_validator,
)


@register_validator(name="qa-relevance-llm-eval", data_type="string")
class QARelevanceLLMEval(Validator):
    """Validates that an answer is relevant to the question asked by asking the
    LLM to self evaluate.

    **Key Properties**

    | Property                      | Description                         |
    | ----------------------------- | ----------------------------------- |
    | Name for `format` attribute   | `qa-relevance-llm-eval`             |
    | Supported data types          | `string`                            |
    | Programmatic fix              | None                                |

    Other parameters: Metadata
        question (str): The original question the llm was given to answer.
    """

    required_metadata_keys = ["question"]

    def __init__(
        self,
        llm_callable: Optional[Callable] = None,
        on_fail: Optional[Callable] = None,
        **kwargs,
    ):
        super().__init__(on_fail, llm_callable=llm_callable, **kwargs)

        if llm_callable is not None and inspect.iscoroutinefunction(llm_callable):
            raise ValueError(
                "QARelevanceLLMEval validator does not support async LLM callables."
            )

        self.llm_callable = (
            llm_callable if llm_callable else get_static_openai_chat_create_func()
        )

    def _selfeval(self, question: str, answer: str) -> Dict:
        from guardrails import Guard

        spec = """
<rail version="0.1">
<output>
    <bool name="relevant" />
</output>

<prompt>
Is the answer below relevant to the question asked?
Question: {question}
Answer: {answer}

Relevant (as a JSON with a single boolean key, "relevant"):\
</prompt>
</rail>
    """.format(
            question=question,
            answer=answer,
        )
        guard = Guard[Dict].from_rail_string(spec)

        response = guard(
            self.llm_callable,  # type: ignore
            max_tokens=10,
            temperature=0.1,
        )
        validated_output = cast(Dict, response.validated_output)
        return validated_output

    def validate(self, value: Any, metadata: Dict) -> ValidationResult:
        if "question" not in metadata:
            raise RuntimeError(
                "qa-relevance-llm-eval validator expects " "`question` key in metadata"
            )

        question = metadata["question"]

        self_evaluation: Dict = self._selfeval(question, value)
        relevant = self_evaluation["relevant"]
        if relevant:
            return PassResult()

        fixed_answer = "No relevant answer found."
        return FailResult(
            error_message=f"The answer {value} is not relevant "
            f"to the question {question}.",
            fix_value=fixed_answer,
        )

    def to_prompt(self, with_keywords: bool = True) -> str:
        return ""
