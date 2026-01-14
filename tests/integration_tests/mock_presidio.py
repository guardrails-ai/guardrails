from typing import List

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


class MockAnalyzerEngine:
    """Mocks the AnalyzerEngine class from presidio-analyzer."""

    def __init__(self) -> None:
        pass


class MockAnonymizerEngine:
    """Mocks the AnonymizerEngine class from presidio-anonymizer."""

    def __init__(self) -> None:
        pass


def mock_anonymize(self, text: str, entities: List[str]) -> str:
    output = None
    if text == "My email address is demo@lol.com, and my phone number is 1234567890":
        if entities == [
            "EMAIL_ADDRESS",
            "PHONE_NUMBER",
        ] or entities == PII_ENTITIES_MAP.get("pii"):
            output = "My email address is <EMAIL_ADDRESS>, and my phone number is <PHONE_NUMBER>"  # noqa
        elif entities == ["EMAIL_ADDRESS"]:
            output = "My email address is <EMAIL_ADDRESS>, and my phone number is 1234567890"
    elif text == "My email address is xyz and my phone number is unavailable.":
        output = text
    return output
