import os
import pytest
from unittest.mock import patch, MagicMock


os.environ["OPENAI_API_KEY"] = "mocked"


# @pytest.fixture(scope="session", autouse=True)
# def mock_tracer():
#     with patch("guardrails.telemetry.common.get_tracer") as mock_get_tracer:
#         mock_get_tracer.return_value = None
#         yield mock_get_tracer


@pytest.fixture(scope="session", autouse=True)
def mock_span():
    with patch("guardrails.telemetry.common.get_span") as mock_get_span:
        mock_get_span.return_value = None
        yield mock_get_span


@pytest.fixture(autouse=True)
def mock_guard_hub_telemetry():
    with patch("guardrails.guard.HubTelemetry") as MockHubTelemetry:
        MockHubTelemetry.return_value = MagicMock()
        MockHubTelemetry.return_value.to_dict = None
        yield MockHubTelemetry


@pytest.fixture(autouse=True)
def mock_validator_base_hub_telemetry():
    with patch("guardrails.validator_base.HubTelemetry") as MockHubTelemetry:
        MockHubTelemetry.return_value = MagicMock()
        MockHubTelemetry.return_value.to_dict = None
        yield MockHubTelemetry


@pytest.fixture(autouse=True)
def mock_runner_hub_telemetry():
    with patch("guardrails.run.runner.HubTelemetry") as MockHubTelemetry:
        MockHubTelemetry.return_value = MagicMock()
        MockHubTelemetry.return_value.to_dict = None
        yield MockHubTelemetry


@pytest.fixture(autouse=True)
def mock_hub_tracing():
    with patch("guardrails.hub_telemetry.hub_tracing.HubTelemetry") as MockHubTelemetry:
        MockHubTelemetry.return_value = MagicMock()
        yield MockHubTelemetry


def pytest_collection_modifyitems(items):
    for item in items:
        if "no_hub_telemetry_mock" in item.keywords:
            item.fixturenames.remove("mock_guard_hub_telemetry")
            item.fixturenames.remove("mock_validator_base_hub_telemetry")
            item.fixturenames.remove("mock_runner_hub_telemetry")
            item.fixturenames.remove("mock_hub_tracing")
