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


@register_validator(name="logical-fallacy-check", data_type="string")
class LogicalFallacyCheck(Validator):
    """Validates that an answer does not contain any logical fallacies that
    arise from RAG on similar entities or otherwise.

    **Key Properties**

    | Property                      | Description                         |
    | ----------------------------- | ----------------------------------- |
    | Name for `format` attribute   | `logical-fallacy-check`             |
    | Supported data types          | `string`                            |
    | Programmatic fix              | None                                |

    Other parameters: Metadata
        question (str): The user prompt or question.
    """

    required_metadata_keys = ["question"]

    def __init__(
        self,
        llm_callable: Optional[Callable] = None,
        on_fail: Optional[Callable] = None,
        **kwargs,
    ):
        super().__init__(
            on_fail,
            llm_callable=llm_callable,
            **kwargs,
        )

        if llm_callable is not None and inspect.iscoroutinefunction(llm_callable):
            raise ValueError(
                "Currently this validator does not support the llm callable value."
            )

        self.llm_callable = (
            llm_callable if llm_callable else get_static_openai_chat_create_func()
        )

    def _selfeval(self, question: str, answer: str) -> Dict:
        from guardrails import Guard

        spec = """
<rail version="0.1">
<output>
    <bool name="logical" />
</output>

<prompt>
Does the answer below contain any logical fallacies in relation to the question asked?
Question: {question}
Answer: {answer}

Relevant (as a JSON with a single boolean key, "logical"):\
</prompt>
</rail>
    """.format(
            question=question,
            answer=answer,
        )
        guard = Guard[Dict].from_rail_string(spec)

        response = guard(
            self.llm_callable,  # type: ignore
            max_tokens=20,
            temperature=0.0,
        )
        validated_output = cast(Dict, response.validated_output)
        return validated_output

    def validate(self, value: Any, metadata: Dict) -> ValidationResult:
        if not metadata:
            # default to value provided via Validator.with_metadata
            metadata = self._metadata

        if "question" not in metadata:
            raise RuntimeError(
                "logical-fallacy-check validator expects "
                "`question` key in metadata or this is an issue with token limitations"
            )

        question = metadata["question"]

        self_evaluation: Dict = self._selfeval(question, value)
        relevant = self_evaluation["logical"]
        if relevant:
            return PassResult()

        fixed_answer = "Generating answer with better logical cohesion..."
        return FailResult(
            error_message=f"The answer {value} contains a logical fallacy in relation "
            f"to the question {question}.",
            fix_value=fixed_answer,
        )

    def to_prompt(self, with_keywords: bool = True) -> str:
        return ""
