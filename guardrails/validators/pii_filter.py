from typing import Any, Callable, Dict, List, Union, cast

from guardrails.validator_base import (
    FailResult,
    PassResult,
    ValidationResult,
    Validator,
    register_validator,
)

try:
    from presidio_analyzer import AnalyzerEngine
    from presidio_anonymizer import AnonymizerEngine
except ImportError:
    AnalyzerEngine = None
    AnonymizerEngine = None


@register_validator(name="pii", data_type="string")
class PIIFilter(Validator):
    """Validates that any text does not contain any PII.

    This validator uses Microsoft's Presidio (https://github.com/microsoft/presidio)
    to detect PII in the text. If PII is detected, the validator will fail with a
    programmatic fix that anonymizes the text. Otherwise, the validator will pass.

    **Key Properties**

    | Property                      | Description                         |
    | ----------------------------- | ----------------------------------- |
    | Name for `format` attribute   | `pii`                               |
    | Supported data types          | `string`                            |
    | Programmatic fix              | Anonymized text with PII filtered   |

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

    def __init__(
        self,
        pii_entities: Union[str, List[str], None] = None,
        on_fail: Union[Callable[..., Any], None] = None,
        **kwargs,
    ):
        if AnalyzerEngine is None or AnonymizerEngine is None:
            raise ImportError(
                "You must install the `presidio-analyzer`, `presidio-anonymizer`"
                "and a spaCy language model to use the PII validator."
                "Refer to https://microsoft.github.io/presidio/installation/"
            )

        super().__init__(on_fail, pii_entities=pii_entities, **kwargs)
        self.pii_entities = pii_entities
        self.pii_analyzer = AnalyzerEngine()
        self.pii_anonymizer = AnonymizerEngine()

    def get_anonymized_text(self, text: str, entities: List[str]) -> str:
        """Analyze and anonymize the text for PII.

        Args:
            text (str): The text to analyze.
            pii_entities (List[str]): The PII entities to filter.

        Returns:
            anonymized_text (str): The anonymized text.
        """
        results = self.pii_analyzer.analyze(text=text, entities=entities, language="en")
        results = cast(List[Any], results)
        anonymized_text = self.pii_anonymizer.anonymize(
            text=text, analyzer_results=results
        ).text
        return anonymized_text

    def validate(self, value: Any, metadata: Dict[str, Any]) -> ValidationResult:
        # Entities to filter passed through metadata take precedence
        pii_entities = metadata.get("pii_entities", self.pii_entities)
        if pii_entities is None:
            raise ValueError(
                "`pii_entities` must be set in order to use the `PIIFilter` validator."
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
            raise ValueError(
                f"`pii_entities` must be one of {pii_keys}" " or a list of strings."
            )

        # Analyze the text, and anonymize it if there is PII
        anonymized_text = self.get_anonymized_text(
            text=value, entities=entities_to_filter
        )

        # If anonymized value text is different from original value, then there is PII
        if anonymized_text != value:
            return FailResult(
                error_message=(
                    f"The following text in your response contains PII:\n{value}"
                ),
                fix_value=anonymized_text,
            )
        return PassResult()
