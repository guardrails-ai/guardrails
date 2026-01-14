import os
from unittest.mock import PropertyMock

import pytest
from fastapi.testclient import TestClient
from fastapi import FastAPI

from guardrails.classes import ValidationOutcome
from guardrails.classes.generic import Stack
from guardrails.classes.history import Call, Iteration
from guardrails.errors import ValidationError

from guardrails_api.app import register_config
from tests.mocks.mock_guard_client import MockGuardStruct
from guardrails_api.api.guards import router as guards_router

# TODO: Should we mock this somehow?
#   Right now it's just empty, but it technically does a file read
register_config()

app = FastAPI()

app.include_router(guards_router)
client = TestClient(app)

MOCK_GUARD_STRING = {
    "id": "mock-guard-id",
    "name": "mock-guard",
    "description": "mock guard description",
    "history": Stack(),
}


@pytest.fixture(autouse=True)
def around_each():
    # Code that will run before the test
    openai_api_key_bak = os.environ.get("OPENAI_API_KEY")
    if openai_api_key_bak:
        del os.environ["OPENAI_API_KEY"]
    yield
    # Code that will run after the test
    if openai_api_key_bak:
        os.environ["OPENAI_API_KEY"] = openai_api_key_bak


def test_guards__get(mocker):
    mock_guard = MockGuardStruct()
    mock_get_guards = mocker.patch(
        "guardrails_api.api.guards.guard_client.get_guards",
        return_value=[mock_guard],
    )
    mocker.patch("guardrails_api.api.guards.collect_telemetry")

    response = client.get("/guards")

    assert mock_get_guards.call_count == 1
    assert response.status_code == 200
    assert response.json() == [MOCK_GUARD_STRING]


def test_guards__post_pg(mocker):
    os.environ["PGHOST"] = "localhost"
    mock_guard = MockGuardStruct()
    mocker.patch(
        "guardrails_api.api.guards.GuardStruct.from_dict",
        return_value=mock_guard,
    )
    mocker.patch(
        "guardrails_api.api.guards.guard_client.create_guard",
        return_value=mock_guard,
    )

    response = client.post("/guards", json=mock_guard.to_dict())

    assert response.status_code == 200
    assert response.json() == MOCK_GUARD_STRING

    del os.environ["PGHOST"]


def test_guards__post_mem(mocker):
    old = None
    if "PGHOST" in os.environ:
        old = os.environ.get("PGHOST")
        del os.environ["PGHOST"]
    mock_guard = MockGuardStruct()

    response = client.post("/guards", json=mock_guard.to_dict())

    assert response.status_code == 501
    assert "Not Implemented" in response.json()["detail"]
    if old:
        os.environ["PGHOST"] = old


def test_guard__get_mem(mocker):
    mock_guard = MockGuardStruct()
    timestamp = "2024-03-04T14:11:42-06:00"
    mock_get_guard = mocker.patch(
        "guardrails_api.api.guards.guard_client.get_guard",
        return_value=mock_guard,
    )

    response = client.get(f"/guards/My%20Guard's%20Name?asOf={timestamp}")

    mock_get_guard.assert_called_once_with("My Guard's Name", timestamp)
    assert response.status_code == 200
    assert response.json() == MOCK_GUARD_STRING


def test_guard__put_pg(mocker):
    os.environ["PGHOST"] = "localhost"
    mock_guard = MockGuardStruct()
    json_guard = {
        "name": "mock-guard",
        "id": "mock-guard-id",
        "description": "mock guard description",
        "history": Stack(),
    }
    mocker.patch(
        "guardrails_api.api.guards.GuardStruct.from_dict",
        return_value=mock_guard,
    )
    mocker.patch(
        "guardrails_api.api.guards.guard_client.upsert_guard",
        return_value=mock_guard,
    )

    response = client.put("/guards/My%20Guard's%20Name", json=json_guard)

    assert response.status_code == 200
    assert response.json() == MOCK_GUARD_STRING
    del os.environ["PGHOST"]


def test_guard__delete_pg(mocker):
    os.environ["PGHOST"] = "localhost"
    mock_guard = MockGuardStruct()
    mock_delete_guard = mocker.patch(
        "guardrails_api.api.guards.guard_client.delete_guard",
        return_value=mock_guard,
    )

    response = client.delete("/guards/my-guard-name")

    mock_delete_guard.assert_called_once_with("my-guard-name")
    assert response.status_code == 200
    assert response.json() == MOCK_GUARD_STRING
    del os.environ["PGHOST"]


def test_validate__parse(mocker):
    os.environ["PGHOST"] = "localhost"
    mock_outcome = ValidationOutcome(
        call_id="mock-call-id",
        raw_llm_output="Hello world!",
        validated_output="Hello world!",
        validation_passed=True,
    )

    mock_parse = mocker.patch.object(MockGuardStruct, "parse")
    mock_parse.return_value = mock_outcome

    mock_guard = MockGuardStruct()
    mock_from_dict = mocker.patch("guardrails_api.api.guards.Guard.from_dict")
    mock_from_dict.return_value = mock_guard

    mock_get_guard = mocker.patch(
        "guardrails_api.api.guards.guard_client.get_guard",
        return_value=mock_guard,
    )

    mock_status = mocker.patch(
        "guardrails.classes.history.call.Call.status", new_callable=PropertyMock
    )
    mock_status.return_value = "pass"
    mock_guard.history = Stack(Call())

    response = client.post(
        "/guards/My%20Guard's%20Name/validate",
        json={"llmOutput": "Hello world!", "args": [1, 2, 3], "some_kwarg": "foo"},
    )

    mock_get_guard.assert_called_once_with("My Guard's Name")
    assert mock_parse.call_count == 1
    mock_parse.assert_called_once_with(
        llm_output="Hello world!",
        num_reasks=None,
        prompt_params={},
        llm_api=None,
        some_kwarg="foo",
        api_key=None,
    )

    assert response.status_code == 200
    assert response.json() == {
        "callId": "mock-call-id",
        "validatedOutput": "Hello world!",
        "validationPassed": True,
        "rawLlmOutput": "Hello world!",
    }

    del os.environ["PGHOST"]


def test_validate__call(mocker):
    os.environ["PGHOST"] = "localhost"
    mock_outcome = ValidationOutcome(
        call_id="mock-call-id",
        raw_llm_output="Hello world!",
        validated_output=None,
        validation_passed=False,
    )

    mock___call__ = mocker.patch.object(MockGuardStruct, "__call__")
    mock___call__.return_value = mock_outcome

    mock_guard = MockGuardStruct()
    mock_from_dict = mocker.patch("guardrails_api.api.guards.Guard.from_dict")
    mock_from_dict.return_value = mock_guard

    mock_get_guard = mocker.patch(
        "guardrails_api.api.guards.guard_client.get_guard",
        return_value=mock_guard,
    )

    mock_status = mocker.patch(
        "guardrails.classes.history.call.Call.status", new_callable=PropertyMock
    )
    mock_status.return_value = "fail"
    mock_guard.history = Stack(Call())

    response = client.post(
        "/guards/My%20Guard's%20Name/validate",
        json={
            "promptParams": {"p1": "bar"},
            "args": [1, 2, 3],
            "some_kwarg": "foo",
            "prompt": "Hello world!",
        },
        headers={"x-openai-api-key": "mock-key"},
    )

    mock_get_guard.assert_called_once_with("My Guard's Name")
    assert mock___call__.call_count == 1
    mock___call__.assert_called_once_with(
        1,
        2,
        3,
        llm_api=None,
        prompt_params={"p1": "bar"},
        num_reasks=None,
        some_kwarg="foo",
        api_key="mock-key",
        prompt="Hello world!",
    )

    assert response.status_code == 200
    assert response.json() == {
        "callId": "mock-call-id",
        "validationPassed": False,
        "validatedOutput": None,
        "rawLlmOutput": "Hello world!",
    }

    del os.environ["PGHOST"]


def test_validate__call_throws_validation_error(mocker):
    os.environ["PGHOST"] = "localhost"
    error = ValidationError("Test guard validation error")
    mock_parse = mocker.patch.object(MockGuardStruct, "__call__")
    mock_parse.side_effect = error

    mock_guard = MockGuardStruct()
    mock_from_dict = mocker.patch("guardrails_api.api.guards.Guard.from_dict")
    mock_from_dict.return_value = mock_guard

    mock_get_guard = mocker.patch(
        "guardrails_api.api.guards.guard_client.get_guard",
        return_value=mock_guard,
    )

    mock_status = mocker.patch(
        "guardrails.classes.history.call.Call.status", new_callable=PropertyMock
    )
    mock_status.return_value = "fail"
    mock_guard.history = Stack(Call())

    response = client.post(
        "/guards/My%20Guard's%20Name/validate",
        json={
            "promptParams": {"p1": "bar"},
            "args": [1, 2, 3],
            "some_kwarg": "foo",
            "prompt": "Hello world!",
        },
    )

    mock_get_guard.assert_called_once_with("My Guard's Name")

    assert response.status_code == 400
    assert response.json() == {"detail": "Test guard validation error"}

    del os.environ["PGHOST"]


def test_openai_v1_chat_completions__raises_404(mocker):
    os.environ["PGHOST"] = "localhost"
    mock_guard = None

    mock_get_guard = mocker.patch(
        "guardrails_api.api.guards.guard_client.get_guard",
        return_value=mock_guard,
    )

    response = client.post(
        "/guards/My%20Guard's%20Name/openai/v1/chat/completions",
        json={
            "messages": [{"role": "user", "content": "Hello world!"}],
        },
        headers={"x-openai-api-key": "mock-key"},
    )

    assert response.status_code == 404
    assert response.json()["detail"] == "A Guard with the name My Guard's Name does not exist!"

    mock_get_guard.assert_called_once_with("My Guard's Name")

    del os.environ["PGHOST"]


def test_openai_v1_chat_completions__call(mocker):
    os.environ["PGHOST"] = "localhost"
    mock_guard = MockGuardStruct()
    mock_outcome = ValidationOutcome(
        call_id="mock-call-id",
        raw_llm_output="Hello world!",
        validated_output="Hello world!",
        validation_passed=False,
    )

    mock___call__ = mocker.patch.object(MockGuardStruct, "__call__")

    mock___call__.return_value = mock_outcome

    mock_from_dict = mocker.patch("guardrails_api.api.guards.Guard.from_dict")
    mock_from_dict.return_value = mock_guard

    mock_get_guard = mocker.patch(
        "guardrails_api.api.guards.guard_client.get_guard",
        return_value=mock_guard,
    )

    mock_status = mocker.patch(
        "guardrails.classes.history.call.Call.status", new_callable=PropertyMock
    )
    mock_status.return_value = "fail"
    mock_call = Call()
    mock_call.iterations = Stack(Iteration("some-id", 1))
    mock_guard.history = Stack(mock_call)

    response = client.post(
        "/guards/My%20Guard's%20Name/openai/v1/chat/completions",
        json={
            "messages": [{"role": "user", "content": "Hello world!"}],
        },
        headers={"x-openai-api-key": "mock-key"},
    )

    mock_get_guard.assert_called_once_with("My Guard's Name")
    assert mock___call__.call_count == 1
    mock___call__.assert_called_once_with(
        num_reasks=0,
        messages=[{"role": "user", "content": "Hello world!"}],
    )

    assert response.status_code == 200
    assert response.json() == {
        "choices": [
            {
                "message": {
                    "content": "Hello world!",
                },
            }
        ],
        "guardrails": {
            "reask": None,
            "validation_passed": False,
            "error": None,
        },
    }

    del os.environ["PGHOST"]
