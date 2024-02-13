import inspect
import os
from typing import Any, Callable, Dict, List, Optional, cast

from guardrails.utils.openai_utils import get_static_openai_chat_create_func
from guardrails.validator_base import (
    FailResult,
    PassResult,
    ValidationResult,
    Validator,
    register_validator,
)


@register_validator(name="saliency-check", data_type="string")
class SaliencyCheck(Validator):
    """Checks that the summary covers the list of topics present in the
    document.

    **Key Properties**

    | Property                      | Description                         |
    | ----------------------------- | ----------------------------------- |
    | Name for `format` attribute   | `saliency-check`                    |
    | Supported data types          | `string`                            |
    | Programmatic fix              | None                                |

    Args:

        docs_dir: Path to the directory containing the documents.
        threshold: Threshold for overlap between topics in document and summary. Defaults to 0.25
    """  # noqa

    def __init__(
        self,
        docs_dir: str,
        llm_callable: Optional[Callable] = None,
        on_fail: Optional[Callable] = None,
        threshold: float = 0.25,
        **kwargs,
    ):
        """Initialize the SalienceCheck validator.

        Args:
            docs_dir: Path to the directory containing the documents.
            on_fail: Function to call when validation fails.
            threshold: Threshold for overlap between topics in document and summary.
        """

        super().__init__(
            on_fail,
            docs_dir=docs_dir,
            llm_callable=llm_callable,
            threshold=threshold,
            **kwargs,
        )

        if llm_callable is not None and inspect.iscoroutinefunction(llm_callable):
            raise ValueError(
                "SaliencyCheck validator does not support async LLM callables."
            )

        self.llm_callable = (
            llm_callable if llm_callable else get_static_openai_chat_create_func()
        )

        self._threshold = threshold

        # Load documents
        self._document_store = {}
        for doc_path in os.listdir(docs_dir):
            with open(os.path.join(docs_dir, doc_path)) as f:
                text = f.read()
            # Precompute topics for each document
            self._document_store[doc_path] = self._get_topics(text)

    @property
    def _topics(self) -> List[str]:
        """Return a list of topics that can be used in the validator."""
        # Merge topics from all documents
        topics = set()
        for doc_topics in self._document_store.values():
            topics.update(doc_topics)
        return list(topics)

    def _get_topics(self, text: str, topics: Optional[List[str]] = None) -> List[str]:
        """Extract topics from a string."""

        from guardrails import Guard

        topics_seed = ""
        if topics is not None:
            topics_seed = (
                "Here's a seed list of topics, select topics from this list"
                " if they are covered in the doc:\n\n" + ", ".join(topics)
            )

        spec = f"""
<rail version="0.1">
<output>
    <list name="topics">
        <string name="topic" description="few words describing the topic in text"/>
    </list>
</output>

<prompt>
Extract a list of topics from the following text:

{text}

{topics_seed}

Return the output as a JSON with a single key "topics" containing a list of topics.

Make sure that topics are relevant to text, and topics are not too specific or general.
</prompt>
</rail>
    """

        guard = Guard.from_rail_string(spec)
        _, validated_output, *rest = guard(llm_api=self.llm_callable)  # type: ignore
        validated_output = cast(Dict, validated_output)
        return validated_output["topics"]

    def validate(self, value: Any, metadata: Dict) -> ValidationResult:
        topics_in_summary = self._get_topics(value, topics=self._topics)

        # Compute overlap between topics in document and summary
        intersection = set(topics_in_summary).intersection(set(self._topics))
        overlap = len(intersection) / len(self._topics)

        if overlap < self._threshold:
            return FailResult(
                error_message=(
                    f"The summary \nSummary: {value}\n does not cover these topics:\n"
                    f"{set(self._topics).difference(intersection)}"
                ),
                fix_value="",
            )

        return PassResult()
