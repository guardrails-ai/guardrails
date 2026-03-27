import subprocess
import sys
from unittest.mock import MagicMock

import pytest

from guardrails.cli.hub.utils import installer_process, PipProcessError


class TestInstallerProcess:
    def test_pip_installer_builds_correct_command(self, mocker):
        mock_run = mocker.patch(
            "guardrails.cli.hub.utils.subprocess.run",
            return_value=MagicMock(stdout="Success", returncode=0),
        )

        result = installer_process(
            "install", "some-package", ["--upgrade"], installer="pip"
        )

        assert result == "Success"
        mock_run.assert_called_once()
        cmd = mock_run.call_args[0][0]
        assert cmd[0] == sys.executable
        assert cmd[1:3] == ["-m", "pip"]
        assert cmd[3] == "install"
        assert "--upgrade" in cmd
        assert "some-package" in cmd

    def test_uv_installer_builds_correct_command(self, mocker):
        mock_run = mocker.patch(
            "guardrails.cli.hub.utils.subprocess.run",
            return_value=MagicMock(stdout="Success", returncode=0),
        )

        result = installer_process(
            "install", "some-package", ["--upgrade"], installer="uv"
        )

        assert result == "Success"
        mock_run.assert_called_once()
        cmd = mock_run.call_args[0][0]
        assert cmd[0] == "uv"
        assert cmd[1] == "pip"
        assert cmd[2] == "install"
        assert "--upgrade" in cmd
        assert "some-package" in cmd

    def test_raises_pip_process_error_on_failure(self, mocker):
        mocker.patch(
            "guardrails.cli.hub.utils.subprocess.run",
            side_effect=subprocess.CalledProcessError(
                1, "pip", output="out", stderr="error msg"
            ),
        )

        with pytest.raises(PipProcessError) as exc_info:
            installer_process("install", "bad-package", installer="pip")

        assert exc_info.value.action == "install"
        assert exc_info.value.package == "bad-package"

    def test_handles_none_stderr_on_failure(self, mocker):
        exc = subprocess.CalledProcessError(1, "pip", output="out")
        exc.stderr = None
        mocker.patch(
            "guardrails.cli.hub.utils.subprocess.run",
            side_effect=exc,
        )

        with pytest.raises(PipProcessError) as exc_info:
            installer_process("install", "bad-package", installer="uv")

        assert exc_info.value.stderr == ""

    def test_no_color_sets_env_var(self, mocker):
        mock_run = mocker.patch(
            "guardrails.cli.hub.utils.subprocess.run",
            return_value=MagicMock(stdout="Success"),
        )

        installer_process("install", "some-package", installer="pip", no_color=True)

        env = mock_run.call_args[1]["env"]
        assert env["NO_COLOR"] == "true"

    @pytest.mark.parametrize("installer", ["uv", "pip"])
    def test_empty_package_not_appended(self, mocker, installer):
        mock_run = mocker.patch(
            "guardrails.cli.hub.utils.subprocess.run",
            return_value=MagicMock(stdout="Success"),
        )

        installer_process("install", "", ["--upgrade"], installer=installer)

        cmd = mock_run.call_args[0][0]
        # Package should not be in command when empty
        assert cmd[-1] == "--upgrade"
