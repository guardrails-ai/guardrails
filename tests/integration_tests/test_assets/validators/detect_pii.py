from typing import Any, Callable, Dict, List, Union
import difflib

from guardrails.validator_base import (
    FailResult,
    PassResult,
    ValidationResult,
    Validator,
    register_validator,
)
from guardrails.validator_base import ErrorSpan


@register_validator(name="guardrails/detect_pii", data_type="string")
class MockDetectPII(Validator):
    """Validates that any text does not contain any PII.

    Instead of using Microsoft Presidio, it accepts a map of PII
    text to their replacements, and performs a simple string replacement.
    For example, if the map is {"John Doe": "REDACTED"}, then the text "John
    Doe is a person" will be replaced with "REDACTED is a person".

    **Key Properties**

    | Property                      | Description                             |
    | ----------------------------- | --------------------------------------- |
    | Name for `format` attribute   | `pii`                                   |
    | Supported data types          | `string`                                |
    | Programmatic fix              | Anonymized text with PII filtered out   |

    Args:
        pii_entities (str | List[str], optional): The PII entities to filter. Must be
            one of `pii` or `spi`. Defaults to None. Can also be set in metadata.
    """

    PII_ENTITIES_MAP = {
        "pii": [
            "EMAIL_ADDRESS",
            "PHONE_NUMBER",
            "DOMAIN_NAME",
            "IP_ADDRESS",
            "DATE_TIME",
            "LOCATION",
            "PERSON",
            "URL",
        ],
        "spi": [
            "CREDIT_CARD",
            "CRYPTO",
            "IBAN_CODE",
            "NRP",
            "MEDICAL_LICENSE",
            "US_BANK_NUMBER",
            "US_DRIVER_LICENSE",
            "US_ITIN",
            "US_PASSPORT",
            "US_SSN",
        ],
    }

    def chunking_function(self, chunk: str):
        """Use a sentence tokenizer to split the chunk into sentences.

        Because using the tokenizer is expensive, we only use it if
        there is a period present in the chunk.
        """
        try:
            import nltk
        except ImportError:
            raise ImportError(
                "nltk is required for sentence splitting. Please install it using `poetry add nltk`"
            )

        # using the sentence tokenizer is expensive
        # we check for a . to avoid wastefully calling the tokenizer
        if "." not in chunk:
            return []
        sentences = nltk.sent_tokenize(chunk)
        if len(sentences) == 0:
            return []
        if len(sentences) == 1:
            sentence = sentences[0].strip()
            # this can still fail if the last chunk ends on the . in an email address
            if sentence[-1] == ".":
                return [sentence, ""]
            else:
                return []

        # return the sentence
        # then the remaining chunks that aren't finished accumulating
        return [sentences[0], "".join(sentences[1:])]

    def __init__(
        self,
        pii_entities: Union[str, List[str], None] = None,
        on_fail: Union[Callable[..., Any], None] = None,
        replace_map: Dict[str, str] = {},
        **kwargs,
    ):
        super().__init__(on_fail, pii_entities=pii_entities, **kwargs)
        self.pii_entities = pii_entities
        self.replace_map = replace_map

    def get_anonymized_text(self, text: str, entities: List[str]) -> str:
        """Analyze and anonymize the text for PII.

        Args:
            text (str): The text to analyze.
            pii_entities (List[str]): The PII entities to filter.

        Returns:
            anonymized_text (str): The anonymized text.
        """
        anonymized_text = text
        # iterate through keys in replace_map
        for key in self.replace_map:
            anonymized_text = anonymized_text.replace(key, self.replace_map[key])
        return anonymized_text

    def validate(self, value: Any, metadata: Dict[str, Any]) -> ValidationResult:
        # Entities to filter passed through metadata take precedence
        pii_entities = metadata.get("pii_entities", self.pii_entities)
        if pii_entities is None:
            raise ValueError(
                "`pii_entities` must be set in order to use the `DetectPII` validator."
                "Add this: `pii_entities=['PERSON', 'PHONE_NUMBER']`"
                "OR pii_entities='pii' or 'spi'"
                "in init or metadata."
            )

        pii_keys = list(self.PII_ENTITIES_MAP.keys())
        # Check that pii_entities is a string OR list of strings
        if isinstance(pii_entities, str):
            # A key to the PII_ENTITIES_MAP
            entities_to_filter = self.PII_ENTITIES_MAP.get(pii_entities, None)
            if entities_to_filter is None:
                raise ValueError(f"`pii_entities` must be one of {pii_keys}")
        elif isinstance(pii_entities, list):
            entities_to_filter = pii_entities
        else:
            raise ValueError(f"`pii_entities` must be one of {pii_keys} or a list of strings.")

        # Analyze the text, and anonymize it if there is PII
        anonymized_text = self.get_anonymized_text(text=value, entities=entities_to_filter)
        if anonymized_text == value:
            return PassResult()

        # TODO: this should be refactored into a helper method in OSS
        # get character indices of differences between two strings
        differ = difflib.Differ()
        diffs = list(differ.compare(value, anonymized_text))
        start_range = None
        diff_ranges = []
        # needs to be tracked separately
        curr_index_in_original = 0
        for i in range(len(diffs)):
            if start_range is not None and diffs[i][0] != "-":
                diff_ranges.append((start_range, curr_index_in_original))
                start_range = None
            if diffs[i][0] == "-":
                if start_range is None:
                    start_range = curr_index_in_original
            if diffs[i][0] != "+":
                curr_index_in_original += 1

        error_spans = []
        for diff_range in diff_ranges:
            error_spans.append(
                ErrorSpan(
                    start=diff_range[0],
                    end=diff_range[1],
                    reason=f"PII detected in {value[diff_range[0] : diff_range[1]]}",
                )
            )

        # If anonymized value text is different from original value, then there is PII
        error_msg = f"The following text in your response contains PII:\n{value}"
        return FailResult(
            error_message=(error_msg),
            fix_value=anonymized_text,
            error_spans=error_spans,
        )
