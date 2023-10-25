from typing import Any, Dict, List, Tuple

SECRETS_CODE_SNIPPET = """
import os
import openai

SECRET_TOKEN = "DUMMY_SECRET_TOKEN_abcdefgh"

ADMIN_CREDENTIALS = {"username": "admin", "password": "dummy_admin_password"}
"""

EXPECTED_SECRETS_CODE_SNIPPET = """
import os
import openai

SECRET_TOKEN = "********"

ADMIN_CREDENTIALS = {"username": "admin", "password": "********"}
"""

NO_SECRETS_CODE_SNIPPET = """
import os
import openai

ADMIN_INFO = {"username": "admin", "country": "United States"}
countries = ["United States", "Canada", "Mexico"]
for country in countries:
    print(country)
    if country == "United States":
        print("Found admin_info for United States")
"""


def mock_get_unique_secrets(self, value: str) -> Tuple[Dict[str, Any], List[str]]:
    lines = value.split("\n")[:-1]
    lines = [line + "\n" for line in lines]
    if value == SECRETS_CODE_SNIPPET:
        unique_secrets = {
            "DUMMY_SECRET_TOKEN_abcdefgh": [5],
            "dummy_admin_password": [7],
        }
    else:
        unique_secrets = {}
    return unique_secrets, lines


class MockDetectSecrets:
    """Mock class for the detect_secrets package."""

    def __init__(self) -> None:
        pass
