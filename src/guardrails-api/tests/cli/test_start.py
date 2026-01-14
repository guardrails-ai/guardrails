from unittest.mock import MagicMock
import os


def test_start(mocker):
    mocker.patch("guardrails_api.cli.start.cli")

    mock_app = MagicMock()
    mock_create_app = mocker.patch("guardrails_api.cli.start.create_app", return_value=mock_app)

    mocker.patch("uvicorn.run")

    from guardrails_api.cli.start import start

    # pg enabled
    os.environ["PGHOST"] = "localhost"
    start("env", "config", 8000)
    os.environ.pop("PGHOST")
    mock_create_app.assert_called_once_with("env", "config", 8000)
