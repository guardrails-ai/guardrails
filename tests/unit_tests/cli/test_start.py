from unittest.mock import MagicMock

from typer.testing import CliRunner

from guardrails.cli.guardrails import guardrails


class TestStart:
    def _make_start_api_mock(self, mocker, api_version="0.3.0"):
        """Set up the common mocks needed for start command tests."""
        mocker.patch("guardrails.cli.start.api_is_installed", return_value=True)
        mocker.patch("guardrails.cli.start.version", return_value=api_version)
        mocker.patch("guardrails.cli.start.version_warnings_if_applicable")
        mocker.patch("guardrails.cli.start.trace_if_enabled")

        mock_start_api = MagicMock()
        mocker.patch.dict(
            "sys.modules",
            {
                "guardrails_api": MagicMock(),
                "guardrails_api.cli": MagicMock(),
                "guardrails_api.cli.start": MagicMock(start=mock_start_api),
            },
        )
        return mock_start_api

    def test_installs_guardrails_api_if_not_present(self, mocker):
        mocker.patch("guardrails.cli.start.api_is_installed", return_value=False)
        mock_installer = mocker.patch("guardrails.cli.start.installer_process")
        mocker.patch("guardrails.cli.start.version", return_value="0.3.0")
        mocker.patch("guardrails.cli.start.version_warnings_if_applicable")
        mocker.patch("guardrails.cli.start.trace_if_enabled")
        mocker.patch.dict(
            "sys.modules",
            {
                "guardrails_api": MagicMock(),
                "guardrails_api.cli": MagicMock(),
                "guardrails_api.cli.start": MagicMock(start=MagicMock()),
            },
        )

        runner = CliRunner()
        runner.invoke(guardrails, ["start"])

        mock_installer.assert_called_once_with("install", "guardrails-api>=0.2.1")

    def test_skips_install_when_guardrails_api_already_present(self, mocker):
        mock_installer = mocker.patch("guardrails.cli.start.installer_process")
        self._make_start_api_mock(mocker, "0.3.0")

        runner = CliRunner()
        runner.invoke(guardrails, ["start"])

        mock_installer.assert_not_called()

    def test_calls_start_api_without_env_override_for_old_api(self, mocker):
        mock_start_api = self._make_start_api_mock(mocker, "0.2.9")

        runner = CliRunner()
        result = runner.invoke(guardrails, ["start"])

        assert result.exit_code == 0
        mock_start_api.assert_called_once_with("", "", 8000)

    def test_calls_start_api_with_env_override_for_new_api(self, mocker):
        mock_start_api = self._make_start_api_mock(mocker, "0.3.0")

        runner = CliRunner()
        result = runner.invoke(guardrails, ["start"])

        assert result.exit_code == 0
        mock_start_api.assert_called_once_with("", "", 8000, False)

    def test_passes_env_override_true_to_new_api(self, mocker):
        mock_start_api = self._make_start_api_mock(mocker, "0.3.0")

        runner = CliRunner()
        result = runner.invoke(guardrails, ["start", "--env-override"])

        assert result.exit_code == 0
        mock_start_api.assert_called_once_with("", "", 8000, True)

    def test_warns_and_ignores_env_override_for_old_api(self, mocker):
        mock_start_api = self._make_start_api_mock(mocker, "0.2.9")
        mock_logger_warning = mocker.patch("guardrails.cli.start.logger.warning")

        runner = CliRunner()
        result = runner.invoke(guardrails, ["start", "--env-override"])

        assert result.exit_code == 0
        mock_logger_warning.assert_called_once()
        warning_msg = mock_logger_warning.call_args[0][0]
        assert (
            "'env_override' is only supported for guardrails-api>=0.3.0" in warning_msg
        )
        assert "0.2.9" in warning_msg
        # env_override is NOT passed to the old API
        mock_start_api.assert_called_once_with("", "", 8000)

    def test_no_warning_when_env_override_false_with_old_api(self, mocker):
        mock_start_api = self._make_start_api_mock(mocker, "0.2.9")
        mock_logger_warning = mocker.patch("guardrails.cli.start.logger.warning")

        runner = CliRunner()
        result = runner.invoke(guardrails, ["start"])

        assert result.exit_code == 0
        mock_logger_warning.assert_not_called()
        mock_start_api.assert_called_once_with("", "", 8000)

    def test_passes_custom_env_file(self, mocker):
        mock_start_api = self._make_start_api_mock(mocker, "0.3.0")

        runner = CliRunner()
        result = runner.invoke(guardrails, ["start", "--env", "custom.env"])

        assert result.exit_code == 0
        mock_start_api.assert_called_once_with("custom.env", "", 8000, False)

    def test_passes_custom_config(self, mocker):
        mock_start_api = self._make_start_api_mock(mocker, "0.3.0")

        runner = CliRunner()
        result = runner.invoke(
            guardrails, ["start", "--config", "guardrails.config.py"]
        )

        assert result.exit_code == 0
        mock_start_api.assert_called_once_with("", "guardrails.config.py", 8000, False)

    def test_passes_custom_port(self, mocker):
        mock_start_api = self._make_start_api_mock(mocker, "0.3.0")

        runner = CliRunner()
        result = runner.invoke(guardrails, ["start", "--port", "9000"])

        assert result.exit_code == 0
        mock_start_api.assert_called_once_with("", "", 9000, False)

    def test_watch_mode_enables_setting(self, mocker):
        from guardrails.settings import settings

        self._make_start_api_mock(mocker, "0.3.0")
        settings._watch_mode_enabled = False

        runner = CliRunner()
        runner.invoke(guardrails, ["start", "--watch"])

        assert settings._watch_mode_enabled is True
        # Cleanup
        settings._watch_mode_enabled = False

    def test_watch_mode_not_set_without_flag(self, mocker):
        from guardrails.settings import settings

        self._make_start_api_mock(mocker, "0.3.0")
        settings._watch_mode_enabled = False

        runner = CliRunner()
        runner.invoke(guardrails, ["start"])

        assert settings._watch_mode_enabled is False

    def test_version_check_major_non_zero_uses_new_signature(self, mocker):
        """A major version != '0' should always use the new API signature."""
        mock_start_api = self._make_start_api_mock(mocker, "1.0.0")

        runner = CliRunner()
        result = runner.invoke(guardrails, ["start", "--env-override"])

        assert result.exit_code == 0
        mock_start_api.assert_called_once_with("", "", 8000, True)

    def test_calls_trace_if_enabled(self, mocker):
        self._make_start_api_mock(mocker, "0.3.0")
        mock_trace = mocker.patch("guardrails.cli.start.trace_if_enabled")

        runner = CliRunner()
        runner.invoke(guardrails, ["start"])

        mock_trace.assert_called_once_with("start")

    def test_calls_version_warnings(self, mocker):
        from guardrails.cli.hub.console import console

        self._make_start_api_mock(mocker, "0.3.0")
        mock_version_warnings = mocker.patch(
            "guardrails.cli.start.version_warnings_if_applicable"
        )

        runner = CliRunner()
        runner.invoke(guardrails, ["start"])

        mock_version_warnings.assert_called_once_with(console)

    def test_logs_starting_server_info(self, mocker):
        self._make_start_api_mock(mocker, "0.3.0")
        mock_logger_info = mocker.patch("guardrails.cli.start.logger.info")

        runner = CliRunner()
        runner.invoke(guardrails, ["start"])

        mock_logger_info.assert_any_call("[INFO]: Starting Guardrails server")
