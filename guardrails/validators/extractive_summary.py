from typing import Any, Callable, Dict, Optional

from guardrails.utils.docs_utils import sentence_split
from guardrails.validator_base import (
    FailResult,
    PassResult,
    ValidationResult,
    Validator,
    register_validator,
)


@register_validator(name="extractive-summary", data_type="string")
class ExtractiveSummary(Validator):
    """Validates that a string is a valid extractive summary of a given
    document.

    This validator does a fuzzy match between the sentences in the
    summary and the sentences in the document. Each sentence in the
    summary must be similar to at least one sentence in the document.
    After the validation, the summary is updated to include the
    sentences from the document that were matched, and the citations for
    those sentences are added to the end of the summary.

    **Key Properties**

    | Property                      | Description                         |
    | ----------------------------- | ----------------------------------- |
    | Name for `format` attribute   | `extractive-summary`                |
    | Supported data types          | `string`                            |
    | Programmatic fix              | Remove any sentences that can not be verified. |

    Args:

        threshold: The minimum fuzz ratio to be considered summarized.  Defaults to 85.

    Other parameters: Metadata

        filepaths (List[str]): A list of strings that specifies the filepaths for any documents that should be used for asserting the summary's similarity.
    """  # noqa

    required_metadata_keys = ["filepaths"]

    def __init__(
        self,
        threshold: int = 85,
        on_fail: Optional[Callable] = None,
        **kwargs,
    ):
        super().__init__(on_fail, threshold=threshold, **kwargs)

        self._threshold = threshold

    def validate(self, value: Any, metadata: Dict) -> ValidationResult:
        """Make sure each sentence was precisely copied from the document."""

        if "filepaths" not in metadata:
            raise RuntimeError(
                "extractive-summary validator expects " "`filepaths` key in metadata"
            )

        filepaths = metadata["filepaths"]

        # Load documents
        store = {}
        for filepath in filepaths:
            with open(filepath) as f:
                doc = f.read()
            store[filepath] = sentence_split(doc)

        try:
            from thefuzz import fuzz  # type: ignore
        except ImportError:
            raise ImportError(
                "`thefuzz` library is required for `extractive-summary` validator. "
                "Please install it with `poetry add thefuzz`."
            )

        # Split the value into sentences.
        sentences = sentence_split(value)

        # Check if any of the sentences in the value match any of the sentences
        # # in the documents.
        unverified = []
        verified = []
        citations = {}

        for id_, sentence in enumerate(sentences):
            highest_ratio = 0
            highest_ratio_doc = None

            # Check fuzzy match against all sentences in all documents
            for doc_path, doc_sentences in store.items():
                for doc_sentence in doc_sentences:
                    ratio = fuzz.ratio(sentence, doc_sentence)
                    if ratio > highest_ratio:
                        highest_ratio = ratio
                        highest_ratio_doc = doc_path

            if highest_ratio < self._threshold:
                unverified.append(sentence)
            else:
                sentence_id = id_ + 1
                citation_id = list(store).index(highest_ratio_doc) + 1

                citations[sentence_id] = citation_id
                verified.append(sentence + f" [{citation_id}]")

        verified_sentences = (
            " ".join(verified)
            + "\n\n"
            + "\n".join(f"[{i + 1}] {s}" for i, s in enumerate(store))
        )

        metadata["summary_with_citations"] = verified_sentences
        metadata["citations"] = citations

        if len(unverified):
            unverified_sentences = "\n".join(
                "- " + s for i, s in enumerate(sentences) if i in unverified
            )
            return FailResult(
                metadata=metadata,
                error_message=(
                    f"The summary \nSummary: {value}\n has sentences\n"
                    f"{unverified_sentences}\n that are not similar to any document."
                ),
                fix_value="\n".join(verified_sentences),
            )

        return PassResult(
            metadata=metadata,
        )
