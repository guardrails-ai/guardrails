import json


from guardrails.cli.hub.migrate import migrate_registry


class TestMigrateRegistry:
    def test_no_hub_init_file(self, tmp_path, mocker):
        mocker.patch(
            "guardrails.cli.hub.migrate.sysconfig.get_paths",
            return_value={"purelib": str(tmp_path / "site-packages")},
        )

        result = migrate_registry(quiet=True)

        assert result == {}

    def test_empty_hub_init(self, tmp_path, mocker):
        site_packages = tmp_path / "site-packages"
        hub_init = site_packages / "guardrails" / "hub" / "__init__.py"
        hub_init.parent.mkdir(parents=True)
        hub_init.write_text("# empty\n")

        mocker.patch(
            "guardrails.cli.hub.migrate.sysconfig.get_paths",
            return_value={"purelib": str(site_packages)},
        )

        result = migrate_registry(quiet=True)

        assert result == {}

    def test_single_validator_migration(self, tmp_path, mocker):
        site_packages = tmp_path / "site-packages"
        hub_init = site_packages / "guardrails" / "hub" / "__init__.py"
        hub_init.parent.mkdir(parents=True)
        hub_init.write_text("from guardrails_grhub_detect_pii import DetectPII\n")

        mocker.patch(
            "guardrails.cli.hub.migrate.sysconfig.get_paths",
            return_value={"purelib": str(site_packages)},
        )
        project_dir = tmp_path / "project"
        project_dir.mkdir()
        mocker.patch(
            "guardrails.cli.hub.migrate.os.getcwd",
            return_value=str(project_dir),
        )

        result = migrate_registry(quiet=True)

        assert "guardrails/detect_pii" in result["validators"]
        entry = result["validators"]["guardrails/detect_pii"]
        assert entry["import_path"] == "guardrails_grhub_detect_pii"
        assert entry["exports"] == ["DetectPII"]
        assert entry["package_name"] == "guardrails-grhub-detect-pii"
        assert "installed_at" in entry

        registry_file = project_dir / ".guardrails" / "hub_registry.json"
        assert registry_file.exists()
        on_disk = json.loads(registry_file.read_text())
        assert on_disk == result

    def test_multiple_validators_migration(self, tmp_path, mocker):
        site_packages = tmp_path / "site-packages"
        hub_init = site_packages / "guardrails" / "hub" / "__init__.py"
        hub_init.parent.mkdir(parents=True)
        hub_init.write_text(
            "from guardrails_grhub_detect_pii import DetectPII\n"
            "from guardrails_ai_grhub_toxic_language import ToxicLanguage\n"
        )

        mocker.patch(
            "guardrails.cli.hub.migrate.sysconfig.get_paths",
            return_value={"purelib": str(site_packages)},
        )
        project_dir = tmp_path / "project"
        project_dir.mkdir()
        mocker.patch(
            "guardrails.cli.hub.migrate.os.getcwd",
            return_value=str(project_dir),
        )

        result = migrate_registry(quiet=True)

        assert len(result["validators"]) == 2
        assert "guardrails/detect_pii" in result["validators"]
        assert "guardrails-ai/toxic_language" in result["validators"]

    def test_multi_export_line(self, tmp_path, mocker):
        site_packages = tmp_path / "site-packages"
        hub_init = site_packages / "guardrails" / "hub" / "__init__.py"
        hub_init.parent.mkdir(parents=True)
        hub_init.write_text(
            "from guardrails_grhub_detect_pii import DetectPII, DetectPIIv2\n"
        )

        mocker.patch(
            "guardrails.cli.hub.migrate.sysconfig.get_paths",
            return_value={"purelib": str(site_packages)},
        )
        project_dir = tmp_path / "project"
        project_dir.mkdir()
        mocker.patch(
            "guardrails.cli.hub.migrate.os.getcwd",
            return_value=str(project_dir),
        )

        result = migrate_registry(quiet=True)

        entry = result["validators"]["guardrails/detect_pii"]
        assert entry["exports"] == ["DetectPII", "DetectPIIv2"]

    def test_skips_comments_and_blank_lines(self, tmp_path, mocker):
        site_packages = tmp_path / "site-packages"
        hub_init = site_packages / "guardrails" / "hub" / "__init__.py"
        hub_init.parent.mkdir(parents=True)
        hub_init.write_text(
            "# Auto-generated file\n"
            "\n"
            "from guardrails_grhub_regex_match import RegexMatch\n"
        )

        mocker.patch(
            "guardrails.cli.hub.migrate.sysconfig.get_paths",
            return_value={"purelib": str(site_packages)},
        )
        project_dir = tmp_path / "project"
        project_dir.mkdir()
        mocker.patch(
            "guardrails.cli.hub.migrate.os.getcwd",
            return_value=str(project_dir),
        )

        result = migrate_registry(quiet=True)

        assert len(result["validators"]) == 1
        assert "guardrails/regex_match" in result["validators"]
