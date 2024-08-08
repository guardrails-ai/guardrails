from unittest.mock import ANY, call
from typer.testing import CliRunner
from guardrails.cli.hub.install import hub_command

import pytest

from guardrails.cli.server.module_manifest import ModuleManifest


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
            quiet=True,
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
            quiet=True,
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
            quiet=True,
            install_local_models_confirm=ANY,
        )

        assert result.exit_code == 0

    def test_install_verbose(self, mocker):
        mock_install = mocker.patch("guardrails.hub.install.install")
        runner = CliRunner()
        result = runner.invoke(
            hub_command, ["install", "hub://guardrails/test-validator", "--verbose"]
        )

        mock_install.assert_called_once_with(
            "hub://guardrails/test-validator",
            install_local_models=None,
            quiet=False,
            install_local_models_confirm=ANY,
        )

        assert result.exit_code == 0


class TestPipProcess:
    def test_no_package_string_format(self, mocker):
        mocker.patch("guardrails.cli.hub.install.os.environ", return_value={})
        mock_logger_debug = mocker.patch("guardrails.cli.hub.utils.logger.debug")

        mock_sys_executable = mocker.patch("guardrails.cli.hub.install.sys.executable")

        mock_subprocess_check_output = mocker.patch(
            "guardrails.cli.hub.install.subprocess.check_output"
        )
        mock_subprocess_check_output.return_value = str.encode("string output")

        from guardrails.cli.hub.install import pip_process

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
        mocker.patch("guardrails.cli.hub.install.os.environ", return_value={})
        mock_logger_debug = mocker.patch("guardrails.cli.hub.install.logger.debug")

        mock_sys_executable = mocker.patch("guardrails.cli.hub.install.sys.executable")

        mock_subprocess_check_output = mocker.patch(
            "guardrails.cli.hub.install.subprocess.check_output"
        )
        mock_subprocess_check_output.return_value = str.encode("json output")

        class MockBytesHeaderParser:
            def parsebytes(self, *args):
                return {"output": "json"}

        mock_bytes_parser = mocker.patch("guardrails.cli.hub.utils.BytesHeaderParser")
        mock_bytes_header_parser = MockBytesHeaderParser()
        mock_bytes_parser.return_value = mock_bytes_header_parser

        from guardrails.cli.hub.install import pip_process

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
        mock_logger_error = mocker.patch("guardrails.cli.hub.install.logger.error")
        mock_logger_debug = mocker.patch("guardrails.cli.hub.install.logger.debug")
        mock_sys_executable = mocker.patch("guardrails.cli.hub.install.sys.executable")
        mock_subprocess_check_output = mocker.patch(
            "guardrails.cli.hub.install.subprocess.check_output"
        )

        from subprocess import CalledProcessError

        mock_subprocess_check_output.side_effect = CalledProcessError(1, "something")

        from guardrails.cli.hub.install import pip_process, sys

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
        mock_logger_debug = mocker.patch("guardrails.cli.hub.install.logger.debug")
        mock_logger_debug.side_effect = error

        mock_logger_error = mocker.patch("guardrails.cli.hub.install.logger.error")

        from guardrails.cli.hub.install import pip_process, sys

        sys_exit_spy = mocker.spy(sys, "exit")

        with pytest.raises(SystemExit):
            pip_process("inspect")

            mock_logger_debug.assert_called_once_with("running pip inspect  ")

            mock_logger_error.assert_called_once_with(
                "An unexpected exception occurred while try to inspect !", error
            )

            sys_exit_spy.assert_called_once_with(1)


def test_get_site_packages_location(mocker):
    mock_pip_process = mocker.patch("guardrails.cli.hub.utils.pip_process")
    mock_pip_process.return_value = {"Location": "/site-pacakges"}

    from guardrails.cli.hub.utils import get_site_packages_location

    response = get_site_packages_location()

    mock_pip_process.assert_called_once_with("show", "pip", format="json")

    assert response == "/site-pacakges"


def test_quiet_install(mocker):
    mock_get_install_url = mocker.patch("guardrails.cli.hub.install.get_install_url")
    mock_get_install_url.return_value = "mock-install-url"

    mock_get_hub_directory = mocker.patch(
        "guardrails.cli.hub.install.get_hub_directory"
    )
    mock_get_hub_directory.return_value = "mock/install/directory"

    mock_pip_process = mocker.patch("guardrails.cli.hub.install.pip_process")
    inspect_report = {
        "installed": [
            {
                "metadata": {
                    "requires_dist": [
                        "rstr",
                        "openai <2",
                        "pydash (>=7.0.6,<8.0.0)",
                        'faiss-cpu (>=1.7.4,<2.0.0) ; extra == "vectordb"',
                    ]
                }
            }
        ]
    }
    mock_pip_process.side_effect = [
        "Sucessfully installed test-validator",
        inspect_report,
        "Sucessfully installed rstr",
        "Sucessfully installed openai<2",
        "Sucessfully installed pydash>=7.0.6,<8.0.0",
    ]

    from guardrails.cli.hub.install import install_hub_module

    manifest = ModuleManifest.from_dict(
        {
            "id": "id",
            "name": "name",
            "author": {"name": "me", "email": "me@me.me"},
            "maintainers": [],
            "repository": {"url": "some-repo"},
            "namespace": "guardrails-ai",
            "package_name": "test-validator",
            "module_name": "validator",
            "exports": ["TestValidator"],
            "tags": {},
        }
    )
    site_packages = "./site-packages"
    install_hub_module(manifest, site_packages, quiet=True)

    mock_get_install_url.assert_called_once_with(manifest)
    mock_get_hub_directory.assert_called_once_with(manifest, site_packages)

    assert mock_pip_process.call_count == 5
    pip_calls = [
        call(
            "install",
            "mock-install-url",
            ["--target=mock/install/directory", "--no-deps", "-q"],
            quiet=True,
        ),
        call(
            "inspect",
            flags=["--path=mock/install/directory"],
            format="json",
            quiet=True,
            no_color=True,
        ),
        call("install", "rstr", quiet=True),
        call("install", "openai<2", quiet=True),
        call("install", "pydash>=7.0.6,<8.0.0", quiet=True),
    ]
    mock_pip_process.assert_has_calls(pip_calls)
