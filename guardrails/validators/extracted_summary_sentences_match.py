import contextvars
import re
from typing import Any, Callable, Dict, Optional

from guardrails.validator_base import (
    FailResult,
    PassResult,
    ValidationResult,
    Validator,
    register_validator,
)


@register_validator(name="extracted-summary-sentences-match", data_type="string")
class ExtractedSummarySentencesMatch(Validator):
    """Validates that the extracted summary sentences match the original text
    by performing a cosine similarity in the embedding space.

    **Key Properties**

    | Property                      | Description                         |
    | ----------------------------- | ----------------------------------- |
    | Name for `format` attribute   | `extracted-summary-sentences-match` |
    | Supported data types          | `string`                            |
    | Programmatic fix              | Remove any sentences that can not be verified. |

    Args:

        threshold: The minimum cosine similarity to be considered similar. Default to 0.7.

    Other parameters: Metadata

        filepaths (List[str]): A list of strings that specifies the filepaths for any documents that should be used for asserting the summary's similarity.
        document_store (DocumentStoreBase, optional): The document store to use during validation. Defaults to EphemeralDocumentStore.
        vector_db (VectorDBBase, optional): A vector database to use for embeddings.  Defaults to Faiss.
        embedding_model (EmbeddingBase, optional): The embeddig model to use. Defaults to OpenAIEmbedding.
    """  # noqa

    required_metadata_keys = ["filepaths"]

    def __init__(
        self,
        threshold: float = 0.7,
        on_fail: Optional[Callable] = None,
        **kwargs: Optional[Dict[str, Any]],
    ):
        super().__init__(on_fail, threshold=threshold, **kwargs)
        # TODO(shreya): Pass embedding_model, vector_db, document_store from spec

        self._threshold = float(threshold)

    @staticmethod
    def _instantiate_store(
        metadata, api_key: Optional[str] = None, api_base: Optional[str] = None
    ):
        if "document_store" in metadata:
            return metadata["document_store"]

        from guardrails.document_store import EphemeralDocumentStore

        if "vector_db" in metadata:
            vector_db = metadata["vector_db"]
        else:
            from guardrails.vectordb import Faiss

            if "embedding_model" in metadata:
                embedding_model = metadata["embedding_model"]
            else:
                from guardrails.embedding import OpenAIEmbedding

                embedding_model = OpenAIEmbedding(api_key=api_key, api_base=api_base)

            vector_db = Faiss.new_flat_ip_index(
                embedding_model.output_dim, embedder=embedding_model
            )

        return EphemeralDocumentStore(vector_db)

    def validate(self, value: Any, metadata: Dict) -> ValidationResult:
        if "filepaths" not in metadata:
            raise RuntimeError(
                "extracted-sentences-summary-match validator expects "
                "`filepaths` key in metadata"
            )
        filepaths = metadata["filepaths"]

        kwargs = {}
        context_copy = contextvars.copy_context()
        for key, context_var in context_copy.items():
            if key.name == "kwargs" and isinstance(kwargs, dict):
                kwargs = context_var
                break

        api_key = kwargs.get("api_key")
        api_base = kwargs.get("api_base")

        store = self._instantiate_store(metadata, api_key, api_base)

        sources = []
        for filepath in filepaths:
            with open(filepath) as f:
                doc = f.read()
                store.add_text(doc, {"path": filepath})
                sources.append(filepath)

        # Split the value into sentences.
        sentences = re.split(r"(?<=[.!?]) +", value)

        # Check if any of the sentences in the value match any of the sentences
        # in the documents.
        unverified = []
        verified = []
        citations = {}
        for id_, sentence in enumerate(sentences):
            page = store.search_with_threshold(sentence, self._threshold)
            if not page or page[0].metadata["path"] not in sources:
                unverified.append(sentence)
            else:
                sentence_id = id_ + 1
                citation_path = page[0].metadata["path"]
                citation_id = sources.index(citation_path) + 1

                citations[sentence_id] = citation_id
                verified.append(sentence + f" [{citation_id}]")

        fixed_summary = (
            " ".join(verified)
            + "\n\n"
            + "\n".join(f"[{i + 1}] {s}" for i, s in enumerate(sources))
        )
        metadata["summary_with_citations"] = fixed_summary
        metadata["citations"] = citations

        if unverified:
            unverified_sentences = "\n".join(unverified)
            return FailResult(
                metadata=metadata,
                error_message=(
                    f"The summary \nSummary: {value}\n has sentences\n"
                    f"{unverified_sentences}\n that are not similar to any document."
                ),
                fix_value=fixed_summary,
            )

        return PassResult(metadata=metadata)

    def to_prompt(self, with_keywords: bool = True) -> str:
        return ""
