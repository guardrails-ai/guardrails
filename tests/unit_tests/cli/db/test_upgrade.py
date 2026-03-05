from importlib.metadata import PackageNotFoundError
from unittest.mock import MagicMock

from typer.testing import CliRunner

from guardrails.cli.db.db import db_command


class TestUpgrade:
    def test_logs_error_when_guardrails_api_not_installed(self, mocker):
        mocker.patch(
            "guardrails.cli.db.upgrade.version",
            side_effect=PackageNotFoundError("guardrails_api"),
        )
        mock_logger_error = mocker.patch("guardrails.cli.db.upgrade.logger.error")

        runner = CliRunner()
        result = runner.invoke(db_command, ["upgrade"])

        assert result.exit_code == 0
        mock_logger_error.assert_called_once_with(
            "[ERROR]: 'db upgrade' requires guardrails-api to be installed."
        )

    def test_logs_error_when_guardrails_api_version_too_old(self, mocker):
        mocker.patch(
            "guardrails.cli.db.upgrade.version",
            return_value="0.2.5",
        )
        mock_logger_error = mocker.patch("guardrails.cli.db.upgrade.logger.error")

        runner = CliRunner()
        result = runner.invoke(db_command, ["upgrade"])

        assert result.exit_code == 0
        mock_logger_error.assert_called_once_with(
            "[ERROR]: 'db upgrade' is only supported for guardrails-api>=0.3.0."
            "  You have guardrails-api==0.2.5."
        )

    def test_logs_error_when_minor_version_is_exactly_below_threshold(self, mocker):
        mocker.patch(
            "guardrails.cli.db.upgrade.version",
            return_value="0.2.99",
        )
        mock_logger_error = mocker.patch("guardrails.cli.db.upgrade.logger.error")

        runner = CliRunner()
        result = runner.invoke(db_command, ["upgrade"])

        assert result.exit_code == 0
        mock_logger_error.assert_called_once()

    def test_delegates_to_guardrails_api_with_defaults(self, mocker):
        mocker.patch(
            "guardrails.cli.db.upgrade.version",
            return_value="0.3.0",
        )
        mock_api_downgrade = MagicMock()
        mocker.patch.dict(
            "sys.modules",
            {
                "guardrails_api": MagicMock(),
                "guardrails_api.cli": MagicMock(),
                "guardrails_api.cli.db": MagicMock(),
                "guardrails_api.cli.db.downgrade": MagicMock(
                    downgrade=mock_api_downgrade
                ),
            },
        )

        runner = CliRunner()
        result = runner.invoke(db_command, ["upgrade"])

        assert result.exit_code == 0
        mock_api_downgrade.assert_called_once_with("head", ".env", False)

    def test_delegates_to_guardrails_api_with_custom_revision(self, mocker):
        mocker.patch(
            "guardrails.cli.db.upgrade.version",
            return_value="0.3.0",
        )
        mock_api_downgrade = MagicMock()
        mocker.patch.dict(
            "sys.modules",
            {
                "guardrails_api": MagicMock(),
                "guardrails_api.cli": MagicMock(),
                "guardrails_api.cli.db": MagicMock(),
                "guardrails_api.cli.db.downgrade": MagicMock(
                    downgrade=mock_api_downgrade
                ),
            },
        )

        runner = CliRunner()
        result = runner.invoke(db_command, ["upgrade", "abc123"])

        assert result.exit_code == 0
        mock_api_downgrade.assert_called_once_with("abc123", ".env", False)

    def test_delegates_to_guardrails_api_with_custom_env_file(self, mocker):
        mocker.patch(
            "guardrails.cli.db.upgrade.version",
            return_value="0.3.0",
        )
        mock_api_downgrade = MagicMock()
        mocker.patch.dict(
            "sys.modules",
            {
                "guardrails_api": MagicMock(),
                "guardrails_api.cli": MagicMock(),
                "guardrails_api.cli.db": MagicMock(),
                "guardrails_api.cli.db.downgrade": MagicMock(
                    downgrade=mock_api_downgrade
                ),
            },
        )

        runner = CliRunner()
        result = runner.invoke(db_command, ["upgrade", "--env", "/custom/.env"])

        assert result.exit_code == 0
        mock_api_downgrade.assert_called_once_with("head", "/custom/.env", False)

    def test_delegates_to_guardrails_api_with_env_override(self, mocker):
        mocker.patch(
            "guardrails.cli.db.upgrade.version",
            return_value="0.3.0",
        )
        mock_api_downgrade = MagicMock()
        mocker.patch.dict(
            "sys.modules",
            {
                "guardrails_api": MagicMock(),
                "guardrails_api.cli": MagicMock(),
                "guardrails_api.cli.db": MagicMock(),
                "guardrails_api.cli.db.downgrade": MagicMock(
                    downgrade=mock_api_downgrade
                ),
            },
        )

        runner = CliRunner()
        result = runner.invoke(db_command, ["upgrade", "--env-override"])

        assert result.exit_code == 0
        mock_api_downgrade.assert_called_once_with("head", ".env", True)

    def test_delegates_to_guardrails_api_with_all_options(self, mocker):
        mocker.patch(
            "guardrails.cli.db.upgrade.version",
            return_value="1.0.0",
        )
        mock_api_downgrade = MagicMock()
        mocker.patch.dict(
            "sys.modules",
            {
                "guardrails_api": MagicMock(),
                "guardrails_api.cli": MagicMock(),
                "guardrails_api.cli.db": MagicMock(),
                "guardrails_api.cli.db.downgrade": MagicMock(
                    downgrade=mock_api_downgrade
                ),
            },
        )

        runner = CliRunner()
        result = runner.invoke(
            db_command,
            ["upgrade", "v2", "--env", "prod.env", "--env-override"],
        )

        assert result.exit_code == 0
        mock_api_downgrade.assert_called_once_with("v2", "prod.env", True)

    def test_does_not_log_error_when_major_version_is_non_zero(self, mocker):
        """Major version != '0' should pass the version check and delegate."""
        mocker.patch(
            "guardrails.cli.db.upgrade.version",
            return_value="1.0.0",
        )
        mock_api_downgrade = MagicMock()
        mocker.patch.dict(
            "sys.modules",
            {
                "guardrails_api": MagicMock(),
                "guardrails_api.cli": MagicMock(),
                "guardrails_api.cli.db": MagicMock(),
                "guardrails_api.cli.db.downgrade": MagicMock(
                    downgrade=mock_api_downgrade
                ),
            },
        )
        mock_logger_error = mocker.patch("guardrails.cli.db.upgrade.logger.error")

        runner = CliRunner()
        runner.invoke(db_command, ["upgrade"])

        mock_logger_error.assert_not_called()
        mock_api_downgrade.assert_called_once()

    def test_default_revision_is_head(self, mocker):
        mocker.patch(
            "guardrails.cli.db.upgrade.version",
            return_value="0.3.0",
        )
        mock_api_downgrade = MagicMock()
        mocker.patch.dict(
            "sys.modules",
            {
                "guardrails_api": MagicMock(),
                "guardrails_api.cli": MagicMock(),
                "guardrails_api.cli.db": MagicMock(),
                "guardrails_api.cli.db.downgrade": MagicMock(
                    downgrade=mock_api_downgrade
                ),
            },
        )

        runner = CliRunner()
        runner.invoke(db_command, ["upgrade"])

        args, _ = mock_api_downgrade.call_args
        assert args[0] == "head"

    def test_version_exactly_030_is_supported(self, mocker):
        mocker.patch(
            "guardrails.cli.db.upgrade.version",
            return_value="0.3.0",
        )
        mock_api_downgrade = MagicMock()
        mocker.patch.dict(
            "sys.modules",
            {
                "guardrails_api": MagicMock(),
                "guardrails_api.cli": MagicMock(),
                "guardrails_api.cli.db": MagicMock(),
                "guardrails_api.cli.db.downgrade": MagicMock(
                    downgrade=mock_api_downgrade
                ),
            },
        )
        mock_logger_error = mocker.patch("guardrails.cli.db.upgrade.logger.error")

        runner = CliRunner()
        runner.invoke(db_command, ["upgrade"])

        mock_logger_error.assert_not_called()
        mock_api_downgrade.assert_called_once()
