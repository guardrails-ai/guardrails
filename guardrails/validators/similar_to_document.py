from typing import Any, Callable, Dict, Optional
from warnings import warn

from guardrails.logger import logger
from guardrails.utils.openai_utils import OpenAIClient
from guardrails.validator_base import (
    VALIDATOR_IMPORT_WARNING,
    VALIDATOR_NAMING,
    FailResult,
    PassResult,
    ValidationResult,
    Validator,
    register_validator,
)

try:
    import numpy as np
except ImportError:
    _HAS_NUMPY = False
else:
    _HAS_NUMPY = True


@register_validator(name="similar-to-document", data_type="string")
class SimilarToDocument(Validator):
    """Validates that a value is similar to the document.

    This validator checks if the value is similar to the document by checking
    the cosine similarity between the value and the document, using an
    embedding.

    **Key Properties**

    | Property                      | Description                       |
    | ----------------------------- | --------------------------------- |
    | Name for `format` attribute   | `similar-to-document`             |
    | Supported data types          | `string`                             |
    | Programmatic fix              | None                              |

    Args:
        document: The document to use for the similarity check.
        threshold: The minimum cosine similarity to be considered similar.  Defaults to 0.7.
        model: The embedding model to use.  Defaults to text-embedding-ada-002.
    """  # noqa

    def __init__(
        self,
        document: str,
        threshold: float = 0.7,
        model: str = "text-embedding-ada-002",
        on_fail: Optional[Callable] = None,
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
        super().__init__(
            on_fail=on_fail, document=document, threshold=threshold, model=model
        )
        if not _HAS_NUMPY:
            raise ImportError(
                f"The {self.__class__.__name__} validator requires the numpy package.\n"
                "`poetry add numpy` to install it."
            )

        self.client = OpenAIClient()

        self._document = document
        embedding_response = self.client.create_embedding(input=[document], model=model)
        embedding = embedding_response[0]  # type: ignore
        self._document_embedding = np.array(embedding)
        self._model = model
        self._threshold = float(threshold)

    @staticmethod
    def cosine_similarity(a: "np.ndarray", b: "np.ndarray") -> float:
        """Calculate the cosine similarity between two vectors.

        Args:
            a: The first vector.
            b: The second vector.

        Returns:
            float: The cosine similarity between the two vectors.
        """
        return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

    def validate(self, value: Any, metadata: Dict) -> ValidationResult:
        logger.debug(f"Validating {value} is similar to document...")

        embedding_response = self.client.create_embedding(
            input=[value], model=self._model
        )

        value_embedding = np.array(embedding_response[0])  # type: ignore

        similarity = self.cosine_similarity(
            self._document_embedding,
            value_embedding,
        )
        if similarity < self._threshold:
            return FailResult(
                error_message=f"Value {value} is not similar enough "
                f"to document {self._document}.",
            )

        return PassResult()

    def to_prompt(self, with_keywords: bool = True) -> str:
        return ""
