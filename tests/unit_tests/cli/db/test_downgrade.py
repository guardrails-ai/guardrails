from importlib.metadata import PackageNotFoundError
from unittest.mock import MagicMock

from typer.testing import CliRunner

from guardrails.cli.db.db import db_command


class TestDowngrade:
    def test_logs_error_when_guardrails_api_not_installed(self, mocker):
        mocker.patch(
            "guardrails.cli.db.downgrade.version",
            side_effect=PackageNotFoundError("guardrails_api"),
        )
        mock_logger_error = mocker.patch("guardrails.cli.db.downgrade.logger.error")

        runner = CliRunner()
        result = runner.invoke(db_command, ["downgrade"])

        assert result.exit_code == 0
        mock_logger_error.assert_called_once_with(
            "[ERROR]: 'db downgrade' requires guardrails-api to be installed."
        )

    def test_logs_error_when_guardrails_api_version_too_old(self, mocker):
        mocker.patch(
            "guardrails.cli.db.downgrade.version",
            return_value="0.2.5",
        )
        mock_logger_error = mocker.patch("guardrails.cli.db.downgrade.logger.error")

        runner = CliRunner()
        result = runner.invoke(db_command, ["downgrade"])

        assert result.exit_code == 0
        mock_logger_error.assert_called_once_with(
            "[ERROR]: 'db downgrade' is only supported for guardrails-api>=0.3.0."
            "  You have guardrails-api==0.2.5."
        )

    def test_logs_error_when_minor_version_is_exactly_below_threshold(self, mocker):
        mocker.patch(
            "guardrails.cli.db.downgrade.version",
            return_value="0.2.99",
        )
        mock_logger_error = mocker.patch("guardrails.cli.db.downgrade.logger.error")

        runner = CliRunner()
        result = runner.invoke(db_command, ["downgrade"])

        assert result.exit_code == 0
        mock_logger_error.assert_called_once()

    def test_delegates_to_guardrails_api_with_defaults(self, mocker):
        mocker.patch(
            "guardrails.cli.db.downgrade.version",
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
        result = runner.invoke(db_command, ["downgrade"])

        assert result.exit_code == 0
        mock_api_downgrade.assert_called_once_with("-1", ".env", False)

    def test_delegates_to_guardrails_api_with_custom_revision(self, mocker):
        mocker.patch(
            "guardrails.cli.db.downgrade.version",
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
        result = runner.invoke(db_command, ["downgrade", "abc123"])

        assert result.exit_code == 0
        mock_api_downgrade.assert_called_once_with("abc123", ".env", False)

    def test_delegates_to_guardrails_api_with_custom_env_file(self, mocker):
        mocker.patch(
            "guardrails.cli.db.downgrade.version",
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
        result = runner.invoke(db_command, ["downgrade", "--env", "/custom/.env"])

        assert result.exit_code == 0
        mock_api_downgrade.assert_called_once_with("-1", "/custom/.env", False)

    def test_delegates_to_guardrails_api_with_env_override(self, mocker):
        mocker.patch(
            "guardrails.cli.db.downgrade.version",
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
        result = runner.invoke(db_command, ["downgrade", "--env-override"])

        assert result.exit_code == 0
        mock_api_downgrade.assert_called_once_with("-1", ".env", True)

    def test_delegates_to_guardrails_api_with_all_options(self, mocker):
        mocker.patch(
            "guardrails.cli.db.downgrade.version",
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
            ["downgrade", "base", "--env", "prod.env", "--env-override"],
        )

        assert result.exit_code == 0
        mock_api_downgrade.assert_called_once_with("base", "prod.env", True)

    def test_does_not_delegate_when_major_version_is_non_zero(self, mocker):
        """Major version != '0' should pass the version check and delegate."""
        mocker.patch(
            "guardrails.cli.db.downgrade.version",
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
        mock_logger_error = mocker.patch("guardrails.cli.db.downgrade.logger.error")

        runner = CliRunner()
        runner.invoke(db_command, ["downgrade"])

        mock_logger_error.assert_not_called()
        mock_api_downgrade.assert_called_once()

    def test_default_revision_is_minus_one(self, mocker):
        mocker.patch(
            "guardrails.cli.db.downgrade.version",
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
        runner.invoke(db_command, ["downgrade"])

        args, _ = mock_api_downgrade.call_args
        assert args[0] == "-1"
