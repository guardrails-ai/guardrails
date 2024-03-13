from typing import Any, Callable, Dict, Optional
from warnings import warn

from guardrails.utils.docs_utils import sentence_split
from guardrails.validator_base import (
    VALIDATOR_IMPORT_WARNING,
    VALIDATOR_NAMING,
    FailResult,
    PassResult,
    ValidationResult,
    Validator,
    register_validator,
)


@register_validator(name="remove-redundant-sentences", data_type="string")
class RemoveRedundantSentences(Validator):
    """Removes redundant sentences from a string.

    This validator removes sentences from a string that are similar to
    other sentences in the string. This is useful for removing
    repetitive sentences from a string.

    **Key Properties**

    | Property                      | Description                         |
    | ----------------------------- | ----------------------------------- |
    | Name for `format` attribute   | `remove-redundant-sentences`        |
    | Supported data types          | `string`                            |
    | Programmatic fix              | Remove any redundant sentences.     |

    Args:

        threshold: The minimum fuzz ratio to be considered redundant.  Defaults to 70.
    """

    def __init__(
        self, threshold: int = 70, on_fail: Optional[Callable] = None, **kwargs
    ):
        class_name = self.__class__.__name__
        if class_name not in VALIDATOR_NAMING:
            warn(
                f"""Validator {class_name} is deprecated and
                will be removed after version 0.5.x.
                """,
                FutureWarning,
            )
        else:
            warn(
                VALIDATOR_IMPORT_WARNING.format(
                    validator_name=class_name,
                    hub_validator_name=VALIDATOR_NAMING.get(class_name)[0],
                    hub_validator_url=VALIDATOR_NAMING.get(class_name)[1],
                ),
                FutureWarning,
            )
        super().__init__(on_fail, threshold=threshold, **kwargs)
        self._threshold = threshold

    def validate(self, value: Any, metadata: Dict) -> ValidationResult:
        """Remove redundant sentences from a string."""

        try:
            from thefuzz import fuzz  # type: ignore
        except ImportError:
            raise ImportError(
                "`thefuzz` library is required for `remove-redundant-sentences` "
                "validator. Please install it with `poetry add thefuzz`."
            )

        # Split the value into sentences.
        sentences = sentence_split(value)
        filtered_sentences = []
        redundant_sentences = []

        sentence = sentences[0]
        other_sentences = sentences[1:]
        while len(other_sentences):
            # Check fuzzy match against all other sentences
            filtered_sentences.append(sentence)
            unique_sentences = []
            for other_sentence in other_sentences:
                ratio = fuzz.ratio(sentence, other_sentence)
                if ratio > self._threshold:
                    redundant_sentences.append(other_sentence)
                else:
                    unique_sentences.append(other_sentence)
            if len(unique_sentences) == 0:
                break
            sentence = unique_sentences[0]
            other_sentences = unique_sentences[1:]

        filtered_summary = " ".join(filtered_sentences)

        if len(redundant_sentences):
            redundant_sentences = "\n".join(redundant_sentences)
            return FailResult(
                error_message=(
                    f"The summary \nSummary: {value}\n has sentences\n"
                    f"{redundant_sentences}\n that are similar to other sentences."
                ),
                fix_value=filtered_summary,
            )

        return PassResult()
