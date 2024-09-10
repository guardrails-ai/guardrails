from unittest.mock import ANY, call
from typer.testing import CliRunner
from guardrails.cli.hub.install import hub_command

import pytest


class TestInstall:
    def test_exits_early_if_uri_is_not_valid(self, mocker):
        mock_logger_error = mocker.patch("guardrails.hub.install.cli_logger.error")

        runner = CliRunner()
        result = runner.invoke(hub_command, ["install", "some-invalid-uri"])

        assert result.exit_code == 1
        mock_logger_error.assert_called_once_with(
            "Invalid URI! The package URI must start with 'hub://'"
        )

    def test_install_local_models__false(self, mocker):
        mock_install = mocker.patch("guardrails.hub.install.install")
        runner = CliRunner()
        result = runner.invoke(
            hub_command,
            ["install", "hub://guardrails/test-validator", "--no-install-local-models"],
        )

        mock_install.assert_called_once_with(
            "hub://guardrails/test-validator",
            install_local_models=False,
            quiet=False,
            install_local_models_confirm=ANY,
        )

        assert result.exit_code == 0

    def test_install_local_models__true(self, mocker):
        mock_install = mocker.patch("guardrails.hub.install.install")
        runner = CliRunner()
        result = runner.invoke(
            hub_command,
            ["install", "hub://guardrails/test-validator", "--install-local-models"],
        )
        mock_install.assert_called_once_with(
            "hub://guardrails/test-validator",
            install_local_models=True,
            quiet=False,
            install_local_models_confirm=ANY,
        )

        assert result.exit_code == 0

    def test_install_local_models__none(self, mocker):
        mock_install = mocker.patch("guardrails.hub.install.install")
        runner = CliRunner()
        result = runner.invoke(
            hub_command,
            ["install", "hub://guardrails/test-validator"],
        )
        mock_install.assert_called_once_with(
            "hub://guardrails/test-validator",
            install_local_models=None,
            quiet=False,
            install_local_models_confirm=ANY,
        )

        assert result.exit_code == 0

    def test_install_quiet(self, mocker):
        mock_install = mocker.patch("guardrails.hub.install.install")
        runner = CliRunner()
        result = runner.invoke(
            hub_command, ["install", "hub://guardrails/test-validator", "--quiet"]
        )

        mock_install.assert_called_once_with(
            "hub://guardrails/test-validator",
            install_local_models=None,
            quiet=True,
            install_local_models_confirm=ANY,
        )

        assert result.exit_code == 0


class TestPipProcess:
    def test_no_package_string_format(self, mocker):
        mocker.patch("guardrails.cli.hub.utils.os.environ", return_value={})
        mock_logger_debug = mocker.patch("guardrails.cli.hub.utils.logger.debug")

        mock_sys_executable = mocker.patch("guardrails.cli.hub.utils.sys.executable")

        mock_subprocess_check_output = mocker.patch(
            "guardrails.cli.hub.utils.subprocess.check_output"
        )
        mock_subprocess_check_output.return_value = str.encode("string output")

        from guardrails.cli.hub.utils import pip_process

        response = pip_process("inspect", flags=["--path=./install-here"])

        assert mock_logger_debug.call_count == 2
        debug_calls = [
            call("running pip inspect --path=./install-here "),
            call("decoding output from pip inspect "),
        ]
        mock_logger_debug.assert_has_calls(debug_calls)

        mock_subprocess_check_output.assert_called_once_with(
            [mock_sys_executable, "-m", "pip", "inspect", "--path=./install-here"],
            env={},
        )

        assert response == "string output"

    def test_json_format(self, mocker):
        mocker.patch("guardrails.cli.hub.utils.os.environ", return_value={})
        mock_logger_debug = mocker.patch("guardrails.cli.hub.utils.logger.debug")

        mock_sys_executable = mocker.patch("guardrails.cli.hub.utils.sys.executable")

        mock_subprocess_check_output = mocker.patch(
            "guardrails.cli.hub.utils.subprocess.check_output"
        )
        mock_subprocess_check_output.return_value = str.encode("json output")

        class MockBytesHeaderParser:
            def parsebytes(self, *args):
                return {"output": "json"}

        mock_bytes_parser = mocker.patch("guardrails.cli.hub.utils.BytesHeaderParser")
        mock_bytes_header_parser = MockBytesHeaderParser()
        mock_bytes_parser.return_value = mock_bytes_header_parser

        from guardrails.cli.hub.utils import pip_process

        response = pip_process("show", "pip", format="json")

        assert mock_logger_debug.call_count == 3
        debug_calls = [
            call("running pip show  pip"),
            call("decoding output from pip show pip"),
            call(
                "JSON parse exception in decoding output from pip show pip. Falling back to accumulating the byte stream"  # noqa
            ),
        ]
        mock_logger_debug.assert_has_calls(debug_calls)

        mock_subprocess_check_output.assert_called_once_with(
            [mock_sys_executable, "-m", "pip", "show", "pip"], env={}
        )

        assert response == {"output": "json"}

    def test_called_process_error(self, mocker):
        mock_logger_error = mocker.patch("guardrails.cli.hub.utils.logger.error")
        mock_logger_debug = mocker.patch("guardrails.cli.hub.utils.logger.debug")
        mock_sys_executable = mocker.patch("guardrails.cli.hub.utils.sys.executable")
        mock_subprocess_check_output = mocker.patch(
            "guardrails.cli.hub.utils.subprocess.check_output"
        )

        from subprocess import CalledProcessError

        mock_subprocess_check_output.side_effect = CalledProcessError(1, "something")

        from guardrails.cli.hub.utils import pip_process, sys

        sys_exit_spy = mocker.spy(sys, "exit")

        with pytest.raises(SystemExit):
            pip_process("inspect")

            mock_logger_debug.assert_called_once_with("running pip inspect  ")

            mock_subprocess_check_output.assert_called_once_with(
                [mock_sys_executable, "-m", "pip", "inspect"]
            )

            mock_logger_error.assert_called_once_with(
                "Failed to inspect \nExit code: 1\nstdout: "
            )

            sys_exit_spy.assert_called_once_with(1)

    def test_other_exception(self, mocker):
        error = ValueError("something went wrong")
        mock_logger_debug = mocker.patch("guardrails.cli.hub.utils.logger.debug")
        mock_logger_debug.side_effect = error

        mock_logger_error = mocker.patch("guardrails.cli.hub.utils.logger.error")

        from guardrails.cli.hub.utils import pip_process, sys

        sys_exit_spy = mocker.spy(sys, "exit")

        with pytest.raises(SystemExit):
            pip_process("inspect")

            mock_logger_debug.assert_called_once_with("running pip inspect  ")

            mock_logger_error.assert_called_once_with(
                "An unexpected exception occurred while try to inspect !", error
            )

            sys_exit_spy.assert_called_once_with(1)

    def test_install_with_upgrade_flag(self, mocker):
        mock_install_hub_module = mocker.patch(
            "guardrails.hub.validator_package_service.ValidatorPackageService.install_hub_module"
        )
        
        runner = CliRunner()
        result = runner.invoke(
            hub_command, ["install", "--upgrade", "hub://guardrails/test-validator"]
        )

        mock_install_hub_module.assert_called_once_with(
            ANY,  
            ANY,  
            quiet=False,
            upgrade=True,
            logger=ANY,
        )

        assert result.exit_code == 0



def test_get_site_packages_location(mocker):
    mock_pip_process = mocker.patch("guardrails.cli.hub.utils.pip_process")
    mock_pip_process.return_value = {"Location": "/site-pacakges"}

    from guardrails.cli.hub.utils import get_site_packages_location

    response = get_site_packages_location()

    mock_pip_process.assert_called_once_with("show", "pip", format="json")

    assert response == "/site-pacakges"
