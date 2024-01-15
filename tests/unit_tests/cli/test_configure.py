from types import TracebackType
from typing import Optional, Type
import pytest

from unittest.mock import call, mock_open

from tests.unit_tests.mocks.mock_file import MockFile


@pytest.mark.parametrize(
        "client_id,client_secret,no_metrics",
        [
            # Note: typer defaults only work through the cli
            # ("mock_client_id", "mock_client_secret", None),
            ("mock_client_id", "mock_client_secret", True),
            ("mock_client_id", "mock_client_secret", False),
        ]
)
def test_configure(mocker, client_id, client_secret, no_metrics):
    mock_save_configuration_file = mocker.patch("guardrails.cli.configure.save_configuration_file")
    mock_logger_info = mocker.patch("guardrails.cli.configure.logger.info")

    from guardrails.cli.configure import configure

    configure(client_id, client_secret, no_metrics)

    mock_logger_info.assert_called_once_with("Configuring...")
    mock_save_configuration_file.assert_called_once_with(
        client_id,
        client_secret,
        no_metrics
    )


def test_configure_prompting(mocker):
    mock_typer_prompt = mocker.patch("typer.prompt")
    mock_typer_prompt.side_effect = ['id', 'secret']
    mock_save_configuration_file = mocker.patch("guardrails.cli.configure.save_configuration_file")
    mock_logger_info = mocker.patch("guardrails.cli.configure.logger.info")

    from guardrails.cli.configure import configure

    configure(None, None, False)

    assert mock_typer_prompt.call_count == 2
    expected_calls = [
        call("Client ID"),
        call("Client secret", hide_input=True)
    ]
    mock_typer_prompt.assert_has_calls(expected_calls)
    mock_logger_info.assert_called_once_with("Configuring...")
    mock_save_configuration_file.assert_called_once_with(
        'id',
        'secret',
        False
    )



def test_save_configuration_file(mocker):
    # TODO: Re-enable this once we move nltk.download calls to individual validator repos.
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

    writelines_spy = mocker.spy(mock_file, "writelines")
    close_spy = mocker.spy(mock_file, "close")

    from guardrails.cli.configure import save_configuration_file

    save_configuration_file("id", "secret", True)

    assert expanduser_mock.called == True
    join_spy.assert_called_once_with("/Home", ".guardrailsrc")

    assert mock_open.call_count == 1
    writelines_spy.assert_called_once_with([
        f"client_id=id{os.linesep}",
        f"client_secret=secret{os.linesep}",
        f"no_metrics=true{os.linesep}",
    ])
    assert close_spy.call_count == 1