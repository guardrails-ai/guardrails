import os
from fastapi.testclient import TestClient
from fastapi import FastAPI
import pytest

from guardrails_api.utils.logger import logger
from tests.mocks.mock_postgres_client import MockPostgresClient

# Assuming you have a similar structure in your FastAPI app
from guardrails_api.api import root


@pytest.fixture
def app():
    app = FastAPI()
    app.include_router(root.router)
    return app


@pytest.fixture
def client(app):
    return TestClient(app)


def test_home(client):
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == "Hello, world!"

    # Check if all expected routes are registered
    routes = [route.path for route in client.app.routes]
    assert "/" in routes
    assert "/health-check" in routes
    assert "/openapi.json" in routes  # This is FastAPI's equivalent to /api-docs
    assert "/docs" in routes


def test_health_check(client, mocker):
    os.environ["PGHOST"] = "localhost"

    mock_pg = MockPostgresClient()
    mock_pg.db.session._set_rows([(1,)])
    mocker.patch("guardrails_api.api.root.PostgresClient", return_value=mock_pg)

    def text_side_effect(query: str):
        return query

    mock_text = mocker.patch("guardrails_api.api.root.text", side_effect=text_side_effect)

    info_spy = mocker.spy(logger, "info")

    response = client.get("/health-check")

    mock_text.assert_called_once_with("SELECT count(datid) FROM pg_stat_activity;")
    assert mock_pg.db.session.queries == ["SELECT count(datid) FROM pg_stat_activity;"]

    info_spy.assert_called_once_with("response: %s", [(1,)])

    assert response.json() == {"status": 200, "message": "Ok"}

    del os.environ["PGHOST"]
