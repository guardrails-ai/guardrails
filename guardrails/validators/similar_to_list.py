from typing import Any, Callable, Dict, Optional

from guardrails.validator_base import (
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


@register_validator(name="similar-to-list", data_type="string")
class SimilarToList(Validator):
    """Validates that a value is similar to a list of previously known values.

    **Key Properties**

    | Property                      | Description                       |
    | ----------------------------- | --------------------------------- |
    | Name for `format` attribute   | `similar-to-list`                 |
    | Supported data types          | `string`                          |
    | Programmatic fix              | None                              |

    Args:
        standard_deviations (int): The number of standard deviations from the mean to check.
        threshold (float): The threshold for the average semantic similarity for strings.

    For integer values, this validator checks whether the value lies
    within 'k' standard deviations of the mean of the previous values.
    (Assumes that the previous values are normally distributed.) For
    string values, this validator checks whether the average semantic
    similarity between the generated value and the previous values is
    less than a threshold.
    """  # noqa

    def __init__(
        self,
        standard_deviations: int = 3,
        threshold: float = 0.1,
        on_fail: Optional[Callable] = None,
        **kwargs,
    ):
        super().__init__(
            on_fail,
            standard_deviations=standard_deviations,
            threshold=threshold,
            **kwargs,
        )
        self._standard_deviations = int(standard_deviations)
        self._threshold = float(threshold)

    def get_semantic_similarity(
        self, text1: str, text2: str, embed_function: Callable
    ) -> float:
        """Get the semantic similarity between two strings.

        Args:
            text1 (str): The first string.
            text2 (str): The second string.
            embed_function (Callable): The embedding function.
        Returns:
            similarity (float): The semantic similarity between the two strings.
        """
        text1_embedding = embed_function(text1)
        text2_embedding = embed_function(text2)
        similarity = 1 - (
            np.dot(text1_embedding, text2_embedding)
            / (np.linalg.norm(text1_embedding) * np.linalg.norm(text2_embedding))
        )
        return similarity

    def validate(self, value: Any, metadata: Dict) -> ValidationResult:
        prev_values = metadata.get("prev_values", [])
        if not prev_values:
            raise ValueError("You must provide a list of previous values in metadata.")

        # Check if np is installed
        if not _HAS_NUMPY:
            raise ValueError(
                "You must install numpy in order to "
                "use the distribution check validator."
            )
        try:
            value = int(value)
            is_int = True
        except ValueError:
            is_int = False

        if is_int:
            # Check whether prev_values are also all integers
            if not all(isinstance(prev_value, int) for prev_value in prev_values):
                raise ValueError(
                    "Both given value and all the previous values must be "
                    "integers in order to use the distribution check validator."
                )

            # Check whether the value lies in a similar distribution as the prev_values
            # Get mean and std of prev_values
            prev_values = np.array(prev_values)
            prev_mean = np.mean(prev_values)  # type: ignore
            prev_std = np.std(prev_values)

            # Check whether the value lies outside specified stds of the mean
            if value < prev_mean - (
                self._standard_deviations * prev_std
            ) or value > prev_mean + (self._standard_deviations * prev_std):
                return FailResult(
                    error_message=(
                        f"The value {value} lies outside of the expected distribution "
                        f"of {prev_mean} +/- {self._standard_deviations * prev_std}."
                    ),
                )
            return PassResult()
        else:
            # Check whether prev_values are also all strings
            if not all(isinstance(prev_value, str) for prev_value in prev_values):
                raise ValueError(
                    "Both given value and all the previous values must be "
                    "strings in order to use the distribution check validator."
                )

            # Check embed model
            embed_function = metadata.get("embed_function", None)
            if embed_function is None:
                raise ValueError(
                    "You must provide `embed_function` in metadata in order to "
                    "check the semantic similarity of the generated string."
                )

            # Check whether the value is semantically similar to the prev_values
            # Get average semantic similarity
            # Lesser the average semantic similarity, more similar the strings are
            avg_semantic_similarity = np.mean(
                np.array(
                    [
                        self.get_semantic_similarity(value, prev_value, embed_function)
                        for prev_value in prev_values
                    ]
                )
            )

            # If average semantic similarity is above the threshold,
            # then the value is not semantically similar to the prev_values
            if avg_semantic_similarity > self._threshold:
                return FailResult(
                    error_message=(
                        f"The value {value} is not semantically similar to the "
                        f"previous values. The average semantic similarity is "
                        f"{avg_semantic_similarity} which is below the threshold of "
                        f"{self._threshold}."
                    ),
                )
            return PassResult()
