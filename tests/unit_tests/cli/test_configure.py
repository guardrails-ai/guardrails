from unittest.mock import call, patch

import pytest
from tests.unit_tests.mocks.mock_file import MockFile


@pytest.mark.parametrize(
    "expected_token, enable_metrics, clear_token",
    [
        ("mock_token", True, False),
        ("mock_token", False, False),
        ("", True, True),
        ("", False, True),
    ],
)
def test_configure(mocker, runner, expected_token, enable_metrics, clear_token):
    mock_save_configuration_file = mocker.patch("guardrails.cli.configure.save_configuration_file")
    mock_logger_info = mocker.patch("guardrails.cli.configure.logger.info")
    mock_get_auth = mocker.patch("guardrails.cli.configure.get_auth")

    CLI_COMMAND = ["configure"]
    CLI_COMMAND_ARGS = []
    CLI_COMMAND_INPUTS = ["mock_token", "mock_input"]

    # Patch sys.stdin with a StringIO object
    from guardrails.cli.guardrails import guardrails

    if enable_metrics:
        CLI_COMMAND_ARGS.append("y")
    else:
        CLI_COMMAND_ARGS.append("n")

    CLI_COMMAND_ARGS.append("")

    if clear_token:
        CLI_COMMAND.append("--clear-token")

    with patch("typer.prompt", side_effect=CLI_COMMAND_INPUTS):
        result = runner.invoke(
            guardrails,
            CLI_COMMAND,
            input="".join([f"{arg}\n" for arg in CLI_COMMAND_ARGS]),
        )

    assert result.exit_code == 0

    expected_calls = [call("Configuration saved.")]

    if clear_token:
        expected_calls.append(call("No token provided. Skipping authentication."))
        assert mock_get_auth.call_count == 0
    else:
        expected_calls.append(call("Validating credentials..."))
        assert mock_get_auth.call_count == 1

    assert mock_logger_info.call_count == 2
    mock_logger_info.assert_has_calls(expected_calls)
    mock_save_configuration_file.assert_called_once_with(expected_token, enable_metrics, True)


def test_save_configuration_file(mocker):
    expanduser_mock = mocker.patch("guardrails.cli.configure.expanduser")
    expanduser_mock.return_value = "/Home"

    rcexpanduser_mock = mocker.patch("guardrails.classes.rc.expanduser")
    rcexpanduser_mock.return_value = "/Home"

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
    assert rcexpanduser_mock.called is True
    join_spy.assert_called_with("/Home", ".guardrailsrc")
    assert join_spy.call_count == 2

    assert mock_open.call_count == 1
    writelines_spy.assert_called_once_with(
        [
            f"id=f49354e0-80c7-4591-81db-cc2f945e5f1e{os.linesep}",
            f"token=token{os.linesep}",
            "enable_metrics=true\n",
            "use_remote_inferencing=true",
        ]
    )
    assert close_spy.call_count == 1
