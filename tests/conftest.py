import os
import pytest
from unittest.mock import patch


os.environ["OPENAI_API_KEY"] = "mocked"


@pytest.fixture(scope="session", autouse=True)
def mock_span():
    with patch("guardrails.telemetry.common.get_span") as mock_get_span:
        mock_get_span.return_value = None
        yield mock_get_span
