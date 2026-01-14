import pytest
import typer


def test_version_callback(mocker):
    mocker.patch("guardrails_api.cli.cli", create=True)
    mock_print = mocker.patch("builtins.print")

    from guardrails_api.cli import version_callback
    from guardrails_api import __version__

    with pytest.raises(typer.Exit):
        version_callback(True)

    mock_print.assert_called_once_with(f"guardrails-api CLI Version: {__version__}")


class TestMain:
    def test_no_args(self, mocker):
        # mocker.patch("guardrails_api.cli.__init__.cli")
        mock_version_callback = mocker.patch("guardrails_api.cli.version_callback")

        from guardrails_api.cli import main

        main()

        assert mock_version_callback.call_count == 0
