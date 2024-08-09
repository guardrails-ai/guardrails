from unittest.mock import mock_open

import pytest
from typer.testing import CliRunner

from guardrails.cli.hub.hub import hub_command


runner = CliRunner()


class TestListCommand:
    @pytest.fixture(autouse=True)
    def setup(self, mocker):
        mocker.patch(
            "guardrails.cli.hub.utils.get_site_packages_location",
            return_value="/test/site-packages",
        )

    def test_list_no_validators_installed(self, mocker):
        mocker.patch("os.path.isfile", return_value=False)

        result = runner.invoke(hub_command, ["list"])

        assert result.exit_code == 0
        assert "No validators installed." in result.output

    def test_list_validators_installed(self, mocker):
        mocker.patch("os.path.isfile", return_value=True)

        init_file_content = """
        from guardrails.hub.guardrails.toxic_language.validator import ToxicLanguage
        from guardrails.hub.guardrails.competitor_check.validator import CompetitorCheck
        """
        mocker.patch("builtins.open", mock_open(read_data=init_file_content))

        result = runner.invoke(hub_command, ["list"])

        assert result.exit_code == 0
        assert "Installed Validators:" in result.output
        assert "- ToxicLanguage" in result.output
        assert "- CompetitorCheck" in result.output

    def test_list_handles_empty_init_file(self, mocker):
        mocker.patch("os.path.isfile", return_value=True)

        mocker.patch("builtins.open", mock_open(read_data=""))

        result = runner.invoke(hub_command, ["list"])

        assert result.exit_code == 0
        assert "No validators installed." in result.output
