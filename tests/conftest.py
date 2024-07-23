import os
from unittest.mock import patch, MagicMock

import pytest


os.environ["OPENAI_API_KEY"] = "mocked"


@pytest.fixture(scope="session", autouse=True)
def mock_tracer():
    with patch("guardrails.utils.telemetry_utils.get_tracer") as mock_get_tracer:
        mock_get_tracer.return_value = None
        yield mock_get_tracer


@pytest.fixture(scope="session", autouse=True)
def mock_span():
    with patch("guardrails.utils.telemetry_utils.get_span") as mock_get_span:
        mock_get_span.return_value = None
        yield mock_get_span


@pytest.fixture(scope="session", autouse=True)
def mock_hub_telemetry():
    with patch("guardrails.utils.hub_telemetry_utils.HubTelemetry") as MockHubTelemetry:
        MockHubTelemetry.return_value = MagicMock
        yield MockHubTelemetry
