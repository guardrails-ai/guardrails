from unittest.mock import call, patch

from tests.unit_tests.mocks.mock_file import MockFile


def test_configure(mocker, runner):
    mock_save_configuration_file = mocker.patch(
        "guardrails.cli.configure.save_configuration_file"
    )
    mock_logger_log = mocker.patch("guardrails.cli.configure.logger.log")

    CLI_COMMAND = ["configure"]
    CLI_COMMAND_ARGS = []
    CLI_COMMAND_INPUTS = ["mock_token", "mock_input"]

    # Patch sys.stdin with a StringIO object
    from guardrails.cli.guardrails import guardrails

    # Answer the "Do you wish to use remote inferencing?" confirm prompt.
    # click >=8.4 aborts on EOF at a confirm prompt instead of falling back
    # to the default, so both prompts must receive explicit input.
    CLI_COMMAND_ARGS.append("y")

    with patch("typer.prompt", side_effect=CLI_COMMAND_INPUTS):
        result = runner.invoke(
            guardrails,
            CLI_COMMAND,
            input="".join([f"{arg}\n" for arg in CLI_COMMAND_ARGS]),
        )

    assert result.exit_code == 0

    expected_calls = [
        call(
            level=35,
            msg="""
        Configuration successful.

        Get started by installing our RegexMatch validator:
        https://guardrailsai.com/hub/validator/guardrails_ai/regex_match

        You can install it by running:
        pip install guardrails-ai-regex-match

        Find more validators at https://guardrailsai.com/hub
        """,
        )
    ]

    assert mock_logger_log.call_count == 1
    mock_logger_log.assert_has_calls(expected_calls)
    mock_save_configuration_file.assert_called_once_with(True)


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

    save_configuration_file(True)

    assert expanduser_mock.called is True
    assert rcexpanduser_mock.called is True
    join_spy.assert_called_with("/Home", ".guardrailsrc")
    assert join_spy.call_count == 2

    assert mock_open.call_count == 1
    writelines_spy.assert_called_once_with(
        [
            f"id=f49354e0-80c7-4591-81db-cc2f945e5f1e{os.linesep}",
            "use_remote_inferencing=true",
        ]
    )
    assert close_spy.call_count == 1
