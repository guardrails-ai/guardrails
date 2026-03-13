import json

from typer.testing import CliRunner

from guardrails.cli.hub.hub import hub_command


runner = CliRunner()


def test_list_from_registry(tmp_path, mocker):
    registry_path = tmp_path / ".guardrails" / "hub_registry.json"
    registry_path.parent.mkdir(parents=True)
    registry = {
        "version": 1,
        "validators": {
            "guardrails/detect-pii": {
                "import_path": "guardrails_grhub_detect_pii",
                "exports": ["DetectPII"],
                "installed_at": "2025-01-01T00:00:00+00:00",
                "package_name": "guardrails-grhub-detect-pii",
            },
            "guardrails/regex-match": {
                "import_path": "guardrails_grhub_regex_match",
                "exports": ["RegexMatch"],
                "installed_at": "2025-01-01T00:00:00+00:00",
                "package_name": "guardrails-grhub-regex-match",
            },
        },
    }
    registry_path.write_text(json.dumps(registry))

    mocker.patch(
        "guardrails.hub.registry.get_registry_path",
        return_value=registry_path,
    )

    result = runner.invoke(hub_command, ["list"])

    assert result.exit_code == 0
    assert "Installed Validators:" in result.output
    assert "guardrails/detect-pii (DetectPII)" in result.output
    assert "guardrails/regex-match (RegexMatch)" in result.output


def test_list_empty_registry(tmp_path, mocker):
    registry_path = tmp_path / ".guardrails" / "hub_registry.json"
    registry_path.parent.mkdir(parents=True)
    registry_path.write_text(json.dumps({"version": 1, "validators": {}}))

    mocker.patch(
        "guardrails.hub.registry.get_registry_path",
        return_value=registry_path,
    )

    result = runner.invoke(hub_command, ["list"])

    assert result.exit_code == 0
    assert "No validators installed." in result.output


def test_list_no_registry_file(tmp_path, mocker):
    registry_path = tmp_path / ".guardrails" / "hub_registry.json"

    mocker.patch(
        "guardrails.hub.registry.get_registry_path",
        return_value=registry_path,
    )

    result = runner.invoke(hub_command, ["list"])

    assert result.exit_code == 0
    assert "No validators installed." in result.output


def test_list_corrupt_registry(tmp_path, mocker):
    registry_path = tmp_path / ".guardrails" / "hub_registry.json"
    registry_path.parent.mkdir(parents=True)
    registry_path.write_text("not valid json{{{")

    mocker.patch(
        "guardrails.hub.registry.get_registry_path",
        return_value=registry_path,
    )

    result = runner.invoke(hub_command, ["list"])

    assert result.exit_code == 0
    assert "No validators installed." in result.output


def test_list_multi_export_validator(tmp_path, mocker):
    registry_path = tmp_path / ".guardrails" / "hub_registry.json"
    registry_path.parent.mkdir(parents=True)
    registry = {
        "version": 1,
        "validators": {
            "guardrails/test-package": {
                "import_path": "guardrails_grhub_test_package",
                "exports": ["Validator", "Helper"],
                "installed_at": "2025-01-01T00:00:00+00:00",
                "package_name": "guardrails-grhub-test-package",
            },
        },
    }
    registry_path.write_text(json.dumps(registry))

    mocker.patch(
        "guardrails.hub.registry.get_registry_path",
        return_value=registry_path,
    )

    result = runner.invoke(hub_command, ["list"])

    assert result.exit_code == 0
    assert "guardrails/test-package (Validator, Helper)" in result.output


def test_list_entry_without_exports_key(tmp_path, mocker):
    """Entry missing exports key should not crash."""
    registry_path = tmp_path / ".guardrails" / "hub_registry.json"
    registry_path.parent.mkdir(parents=True)
    registry = {
        "version": 1,
        "validators": {
            "guardrails/test": {
                "import_path": "guardrails_grhub_test",
                "installed_at": "2025-01-01T00:00:00+00:00",
                "package_name": "guardrails-grhub-test",
                "exports": [],
            }
        },
    }
    registry_path.write_text(json.dumps(registry))

    mocker.patch(
        "guardrails.hub.registry.get_registry_path",
        return_value=registry_path,
    )

    result = runner.invoke(hub_command, ["list"])

    assert result.exit_code == 0
    assert "guardrails/test ()" in result.output
