from unittest.mock import call

import pytest

from tests.unit_tests.mocks.mock_file import MockFile


@pytest.mark.parametrize(
    "token,no_metrics",
    [
        # Note: typer defaults only work through the cli
        # ("mock_client_id", "mock_client_secret", None),
        ("mock_token", True),
        ("mock_token", False),
    ],
)
def test_configure(mocker, token, no_metrics):
    mock_save_configuration_file = mocker.patch(
        "guardrails.cli.configure.save_configuration_file"
    )
    mock_logger_info = mocker.patch("guardrails.cli.configure.logger.info")
    mock_get_auth = mocker.patch("guardrails.cli.configure.get_auth")

    from guardrails.cli.configure import configure

    configure(token, no_metrics)

    assert mock_logger_info.call_count == 2
    expected_calls = [call("Validating credentials..."), call("Configuration saved.")]
    mock_logger_info.assert_has_calls(expected_calls)

    mock_save_configuration_file.assert_called_once_with(token, no_metrics)

    assert mock_get_auth.call_count == 1


def test_save_configuration_file(mocker):
    # TODO: Re-enable this once we move nltk.download calls to individual validator repos.  # noqa
    # Right now, it fires during our import chain, causing this to blow up
    mocker.patch("nltk.data.find")
    mocker.patch("nltk.download")

    expanduser_mock = mocker.patch("guardrails.cli.configure.expanduser")
    expanduser_mock.return_value = "/Home"

    import os

    join_spy = mocker.spy(os.path, "join")

    mock_file = MockFile()
    mock_open = mocker.patch("guardrails.cli.configure.open")
    mock_open.return_value = mock_file

    mock_uuid = mocker.patch("guardrails.cli.configure.uuid.uuid4")
    mock_uuid.return_value = "f49354e0-80c7-4591-81db-cc2f945e5f1e"

    writelines_spy = mocker.spy(mock_file, "writelines")
    close_spy = mocker.spy(mock_file, "close")

    from guardrails.cli.configure import save_configuration_file

    save_configuration_file("token", True)

    assert expanduser_mock.called is True
    join_spy.assert_called_once_with("/Home", ".guardrailsrc")

    assert mock_open.call_count == 1
    writelines_spy.assert_called_once_with(
        [
            f"id=f49354e0-80c7-4591-81db-cc2f945e5f1e{os.linesep}",
            f"token=token{os.linesep}",
            "enable_metrics=true",
        ]
    )
    assert close_spy.call_count == 1
