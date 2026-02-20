import json
import os
from pathlib import Path
from typing import cast
import pytest
import sys
from unittest.mock import call, patch, MagicMock

from guardrails_hub_types import Manifest
from guardrails.cli.hub.utils import PipProcessError
from guardrails.hub.validator_package_service import (
    FailedToLocateModule,
    ValidatorPackageService,
    InvalidHubInstallURL,
    _GUARDRAILS_INSTALLER_ENV,
)
from tests.unit_tests.mocks.mock_file import MockFile


class TestGetModulePath:
    @patch.dict("sys.modules")
    @patch("guardrails.hub.validator_package_service.importlib")
    def test_get_module_path_package_in_sys_modules(self, mock_importlib):
        sys.modules["pip"] = MagicMock()
        sys.modules["pip"].__path__ = ["/fake/site-packages/pip"]

        module_path = ValidatorPackageService.get_module_path("pip")
        assert module_path == "/fake/site-packages/pip"

    @patch("guardrails.hub.validator_package_service.importlib")
    @patch.dict("sys.modules")
    def test_get_module_path_package_not_in_sys_modules(self, mock_importlib):
        sys.modules.pop("pip", None)

        mock_module = MagicMock()
        mock_module.__path__ = ["/fake/site-packages/pip"]
        mock_importlib.import_module.return_value = mock_module

        module_path = ValidatorPackageService.get_module_path("pip")
        assert module_path == "/fake/site-packages/pip"

    @patch.dict("sys.modules")
    @patch("guardrails.hub.validator_package_service.importlib")
    def test_get_module_path_failed_to_locate_module(self, mock_importlib):
        sys.modules.pop("pip", None)

        with pytest.raises(FailedToLocateModule):
            ValidatorPackageService.get_module_path("invalid-module")


class TestAddToHubInits:
    def test_closes_early_if_already_added(self, mocker):
        manifest = Manifest.from_dict(
            {
                "id": "guardrails-ai/id",
                "name": "name",
                "author": {"name": "me", "email": "me@me.me"},
                "maintainers": [],
                "repository": {"url": "some-repo"},
                "namespace": "guardrails-ai",
                "packageName": "test-validator",
                "moduleName": "validator",
                "description": "description",
                "exports": ["TestValidator", "helper"],
                "tags": {},
            }
        )
        site_packages = "./site-packages"

        hub_init_file = MockFile()
        mock_open = mocker.patch("guardrails.hub.validator_package_service.open")
        mock_open.side_effect = [hub_init_file]

        mock_hub_read = mocker.patch.object(hub_init_file, "read")
        mock_hub_read.return_value = (
            "from guardrails_ai_grhub_id import helper, TestValidator"  # noqa
        )

        hub_seek_spy = mocker.spy(hub_init_file, "seek")
        hub_write_spy = mocker.spy(hub_init_file, "write")
        hub_close_spy = mocker.spy(hub_init_file, "close")

        from guardrails.hub.validator_package_service import ValidatorPackageService

        manifest = cast(Manifest, manifest)
        ValidatorPackageService.add_to_hub_inits(manifest, site_packages)

        assert mock_open.call_count == 1
        open_calls = [
            call("./site-packages/guardrails/hub/__init__.py", "a+"),
        ]
        mock_open.assert_has_calls(open_calls)

        assert hub_seek_spy.call_count == 1
        assert mock_hub_read.call_count == 1
        assert hub_write_spy.call_count == 0
        assert hub_close_spy.call_count == 1

    def test_appends_import_line_if_not_present(self, mocker):
        manifest = Manifest.from_dict(
            {
                "id": "guardrails-ai/id",
                "name": "name",
                "author": {"name": "me", "email": "me@me.me"},
                "maintainers": [],
                "repository": {"url": "some-repo"},
                "namespace": "guardrails-ai",
                "packageName": "test-validator",
                "moduleName": "validator",
                "description": "description",
                "exports": ["TestValidator"],
                "tags": {},
            }
        )
        site_packages = "./site-packages"

        hub_init_file = MockFile()
        mock_open = mocker.patch("guardrails.hub.validator_package_service.open")
        mock_open.side_effect = [hub_init_file]

        mock_hub_read = mocker.patch.object(hub_init_file, "read")
        mock_hub_read.return_value = "from guardrails.hub.other_org.other_validator.validator import OtherValidator"  # noqa

        hub_seek_spy = mocker.spy(hub_init_file, "seek")
        hub_write_spy = mocker.spy(hub_init_file, "write")
        hub_close_spy = mocker.spy(hub_init_file, "close")

        from guardrails.hub.validator_package_service import ValidatorPackageService

        manifest = cast(Manifest, manifest)
        ValidatorPackageService.add_to_hub_inits(manifest, site_packages)

        assert mock_open.call_count == 1
        open_calls = [
            call("./site-packages/guardrails/hub/__init__.py", "a+"),
        ]
        mock_open.assert_has_calls(open_calls)

        assert hub_seek_spy.call_count == 2
        hub_seek_calls = [call(0, 0), call(0, 2)]
        hub_seek_spy.assert_has_calls(hub_seek_calls)

        assert mock_hub_read.call_count == 1

        assert hub_write_spy.call_count == 2
        hub_write_calls = [
            call("\n"),
            call(
                "from guardrails_ai_grhub_id import TestValidator"  # noqa
            ),
        ]
        hub_write_spy.assert_has_calls(hub_write_calls)

        assert hub_close_spy.call_count == 1

    def test_creates_namespace_init_if_not_exists(self, mocker):
        manifest = Manifest.from_dict(
            {
                "id": "guardrails-ai/id",
                "name": "name",
                "author": {"name": "me", "email": "me@me.me"},
                "maintainers": [],
                "repository": {"url": "some-repo"},
                "namespace": "guardrails-ai",
                "packageName": "test-validator",
                "moduleName": "validator",
                "description": "description",
                "exports": ["TestValidator"],
                "tags": {},
            }
        )
        site_packages = "./site-packages"

        hub_init_file = MockFile()
        mock_open = mocker.patch("guardrails.hub.validator_package_service.open")
        mock_open.side_effect = [hub_init_file]

        mock_hub_read = mocker.patch.object(hub_init_file, "read")
        mock_hub_read.return_value = "from guardrails_ai_grhub_id import TestValidator"  # noqa

        mock_is_file = mocker.patch(
            "guardrails.hub.validator_package_service.os.path.isfile"
        )
        mock_is_file.return_value = False

        from guardrails.hub.validator_package_service import ValidatorPackageService

        manifest = cast(Manifest, manifest)
        ValidatorPackageService.add_to_hub_inits(manifest, site_packages)

        assert mock_open.call_count == 1
        open_calls = [
            call("./site-packages/guardrails/hub/__init__.py", "a+"),
        ]
        mock_open.assert_has_calls(open_calls)


class TestReloadModule:
    @patch("guardrails.hub.validator_package_service.importlib")
    @patch.dict("sys.modules")
    def test_reload_module__guardrails_hub_reload_if_in_sys_modules(
        self, mock_importlib
    ):
        sys.modules["guardrails.hub"] = MagicMock()
        mock_module = MagicMock()
        mock_importlib.reload.return_value = mock_module
        ValidatorPackageService.reload_module("guardrails.hub")
        mock_importlib.reload.assert_called_once_with(sys.modules["guardrails.hub"])

    @patch("guardrails.hub.validator_package_service.importlib")
    @patch.dict("sys.modules")
    def test_reload_module__guardrails_hub_reload_if_not_in_sys_modules(
        self, mock_importlib
    ):
        sys.modules.pop("guardrails.hub", None)
        ValidatorPackageService.reload_module("guardrails.hub")
        # assert not called
        mock_importlib.reload.assert_not_called()

    @patch("guardrails.hub.validator_package_service.importlib")
    @patch.dict("sys.modules")
    def test_reload_module__module_not_found(self, mock_importlib):
        mock_importlib.import_module.side_effect = ModuleNotFoundError(
            "Module not found"
        )

        with pytest.raises(ModuleNotFoundError):
            ValidatorPackageService.reload_module(
                "guardrails.hub.guardrails.contains_string.validator"
            )

    @patch("guardrails.hub.validator_package_service.importlib")
    @patch.dict("sys.modules")
    def test_reload_module__unexpected_exception(self, mock_importlib):
        mock_importlib.import_module.side_effect = Exception("Unexpected exception")

        with pytest.raises(Exception):
            ValidatorPackageService.reload_module(
                "guardrails.hub.guardrails.contains_string.validator"
            )

    @patch("guardrails.hub.validator_package_service.importlib")
    @patch.dict("sys.modules")
    def test_reload_module__module_already_imported(self, mock_importlib):
        mock_validator_module = MagicMock()
        sys.modules["guardrails.hub.guardrails.contains_string.validator"] = (
            mock_validator_module
        )

        reloaded_module = ValidatorPackageService.reload_module(
            "guardrails.hub.guardrails.contains_string.validator"
        )

        assert reloaded_module == mock_validator_module

    @patch("guardrails.hub.validator_package_service.importlib")
    @patch.dict("sys.modules")
    def test_reload_module__module_not_imported(self, mock_importlib):
        mock_module = MagicMock()
        mock_importlib.import_module.return_value = mock_module

        sys.modules.pop("guardrails.hub.guardrails.contains_string.validator", None)
        reloaded_module = ValidatorPackageService.reload_module(
            "guardrails.hub.guardrails.contains_string.validator"
        )

        assert reloaded_module == mock_module


class TestRunPostInstall:
    @pytest.mark.parametrize(
        "manifest",
        [
            Manifest.from_dict(
                {
                    "id": "guardrails-ai/id",
                    "name": "name",
                    "author": {"name": "me", "email": "me@me.me"},
                    "maintainers": [],
                    "repository": {"url": "some-repo"},
                    "namespace": "guardrails-ai",
                    "packageName": "test-validator",
                    "moduleName": "validator",
                    "description": "description",
                    "exports": ["TestValidator"],
                    "tags": {},
                }
            ),
            Manifest.from_dict(
                {
                    "id": "guardrails-ai/id",
                    "name": "name",
                    "author": {"name": "me", "email": "me@me.me"},
                    "maintainers": [],
                    "repository": {"url": "some-repo"},
                    "namespace": "guardrails-ai",
                    "packageName": "test-validator",
                    "moduleName": "validator",
                    "description": "description",
                    "exports": ["TestValidator"],
                    "tags": {},
                    "post_install": "",
                }
            ),
        ],
    )
    def test_does_not_run_if_no_script(self, mocker, manifest):
        mock_subprocess_check_output = mocker.patch(
            "guardrails.hub.validator_package_service.subprocess.check_output"
        )
        from guardrails.hub.validator_package_service import ValidatorPackageService

        ValidatorPackageService.run_post_install(manifest, "./site_packages")

        assert mock_subprocess_check_output.call_count == 0

    def test_runs_script_if_exists(self, mocker):
        mock_subprocess_check_output = mocker.patch(
            "guardrails.hub.validator_package_service.subprocess.check_output"
        )
        mock_sys_executable = mocker.patch(
            "guardrails.hub.validator_package_service.sys.executable"
        )
        mock_isfile = mocker.patch(
            "guardrails.hub.validator_package_service.os.path.isfile"
        )
        mock_isfile.return_value = True
        from guardrails.hub.validator_package_service import ValidatorPackageService

        manifest = Manifest.from_dict(
            {
                "id": "guardrails-ai/id",
                "name": "name",
                "author": {"name": "me", "email": "me@me.me"},
                "maintainers": [],
                "repository": {"url": "some-repo"},
                "namespace": "guardrails-ai",
                "packageName": "test-validator",
                "moduleName": "validator",
                "description": "description",
                "exports": ["TestValidator"],
                "tags": {},
                "postInstall": "post_install.py",
            }
        )

        manifest = cast(Manifest, manifest)
        ValidatorPackageService.run_post_install(manifest, "./site_packages")

        assert mock_subprocess_check_output.call_count == 1
        mock_subprocess_check_output.assert_called_once_with(
            [
                mock_sys_executable,
                "./site_packages/guardrails_ai_grhub_id/post_install.py",  # noqa
            ]
        )


class TestValidatorPackageService:
    def setup_method(self):
        self.manifest = Manifest.from_dict(
            {
                "id": "guardrails/id",
                "name": "name",
                "author": {"name": "me", "email": "me@me.me"},
                "maintainers": [],
                "repository": {"url": "some-repo"},
                "namespace": "guardrails",
                "packageName": "test-validator",
                "moduleName": "test_validator",
                "description": "description",
                "exports": ["TestValidator"],
                "tags": {"has_guardrails_endpoint": False},
            }
        )
        self.site_packages = "./.venv/lib/python3.X/site-packages"

    @patch("guardrails.hub.validator_package_service.get_validator_manifest")
    @patch(
        "guardrails.hub.validator_package_service.ValidatorPackageService.get_site_packages_location"
    )
    def test_get_manifest_and_site_packages(
        self, mock_get_site_packages_location, mock_get_validator_manifest
    ):
        # Setup
        mock_get_validator_manifest.return_value = self.manifest
        mock_get_site_packages_location.return_value = self.site_packages

        # Test
        manifest, site_packages = (
            ValidatorPackageService.get_manifest_and_site_packages("test-module")
        )

        # Assert
        assert manifest == self.manifest
        assert site_packages == self.site_packages
        mock_get_validator_manifest.assert_called_once_with("test-module")
        mock_get_site_packages_location.assert_called_once()

    @patch("guardrails.hub.validator_package_service.sysconfig.get_paths")
    def test_get_site_packages_location(self, mock_get_paths):
        mock_get_paths.return_value = {"purelib": "/fake/site-packages"}
        site_packages_path = ValidatorPackageService.get_site_packages_location()
        assert site_packages_path == "/fake/site-packages"

    @patch(
        "guardrails.hub.validator_package_service.ValidatorPackageService.reload_module"
    )
    def test_get_validator_from_manifest(self, mock_reload_module):
        mock_validator_module = MagicMock()
        mock_reload_module.return_value = mock_validator_module

        manifest = cast(Manifest, self.manifest)
        ValidatorPackageService.get_validator_from_manifest(manifest)

        mock_reload_module.assert_called_once_with("guardrails_grhub_id")

    def test_get_module_name_valid(self):
        module_name, module_version = ValidatorPackageService.get_validator_id(
            "hub://test-module>=1.0.0"
        )
        assert module_name == "test-module"
        assert module_version == ">=1.0.0"

    def test_get_module_name_invalid(self):
        with pytest.raises(InvalidHubInstallURL):
            ValidatorPackageService.get_validator_id("invalid-uri")

    def test_install_hub_module_when_exception(self, mocker):
        mock_installer = mocker.patch(
            "guardrails.hub.validator_package_service.installer_process"
        )
        mock_settings = mocker.patch(
            "guardrails.hub.validator_package_service.settings"
        )
        mock_settings.rc.token = "mock-token"
        mocker.patch.object(
            ValidatorPackageService, "detect_installer", return_value="pip"
        )

        mock_installer.side_effect = [Exception()]

        manifest = Manifest.from_dict(
            {
                "id": "guardrails-ai/id",
                "name": "name",
                "author": {"name": "me", "email": "me@me.me"},
                "maintainers": [],
                "repository": {"url": "some-repo"},
                "namespace": "guardrails-ai",
                "packageName": "test-validator",
                "moduleName": "validator",
                "description": "description",
                "exports": ["TestValidator"],
                "tags": {},
            }
        )
        manifest = cast(Manifest, manifest)

        with pytest.raises(Exception):
            ValidatorPackageService.install_hub_module(manifest.id)

        assert mock_installer.call_count == 1

    def test_install_hub_module_when_no_validators_extras(self, mocker):
        mock_installer = mocker.patch(
            "guardrails.hub.validator_package_service.installer_process"
        )
        mock_settings = mocker.patch(
            "guardrails.hub.validator_package_service.settings"
        )
        mock_settings.rc.token = "mock-token"
        mocker.patch.object(
            ValidatorPackageService, "detect_installer", return_value="pip"
        )

        mock_installer.side_effect = [
            PipProcessError("install", "guardrails-ai-grhub-id"),
            "Sucessfully installed guardrails-ai-grhub-id",
        ]

        manifest = Manifest.from_dict(
            {
                "id": "guardrails-ai/id",
                "name": "name",
                "author": {"name": "me", "email": "me@me.me"},
                "maintainers": [],
                "repository": {"url": "some-repo"},
                "namespace": "guardrails-ai",
                "packageName": "test-validator",
                "moduleName": "validator",
                "description": "description",
                "exports": ["TestValidator"],
                "tags": {},
            }
        )
        manifest = cast(Manifest, manifest)
        ValidatorPackageService.install_hub_module(manifest.id)

        assert mock_installer.call_count == 2
        installer_calls = [
            call(
                "install",
                "guardrails-ai-grhub-id[validators]",
                [
                    "--index-url=https://__token__:mock-token@pypi.guardrailsai.com/simple",
                    "--extra-index-url=https://pypi.org/simple",
                ],
                quiet=False,
                installer="pip",
            ),
            call(
                "install",
                "guardrails-ai-grhub-id",
                [
                    "--index-url=https://__token__:mock-token@pypi.guardrailsai.com/simple",
                    "--extra-index-url=https://pypi.org/simple",
                ],
                quiet=False,
                installer="pip",
            ),
        ]
        mock_installer.assert_has_calls(installer_calls)

    def test_install_hub_module(self, mocker):
        mock_installer = mocker.patch(
            "guardrails.hub.validator_package_service.installer_process"
        )
        mock_settings = mocker.patch(
            "guardrails.hub.validator_package_service.settings"
        )
        mock_settings.rc.token = "mock-token"
        mocker.patch.object(
            ValidatorPackageService, "detect_installer", return_value="pip"
        )

        mock_installer.return_value = "Sucessfully installed test-validator"

        manifest = Manifest.from_dict(
            {
                "id": "guardrails-ai/id",
                "name": "name",
                "author": {"name": "me", "email": "me@me.me"},
                "maintainers": [],
                "repository": {"url": "some-repo"},
                "namespace": "guardrails-ai",
                "packageName": "test-validator",
                "moduleName": "validator",
                "description": "description",
                "exports": ["TestValidator"],
                "tags": {},
            }
        )
        manifest = cast(Manifest, manifest)
        ValidatorPackageService.install_hub_module(manifest.id)

        assert mock_installer.call_count == 1
        installer_calls = [
            call(
                "install",
                "guardrails-ai-grhub-id[validators]",
                [
                    "--index-url=https://__token__:mock-token@pypi.guardrailsai.com/simple",
                    "--extra-index-url=https://pypi.org/simple",
                ],
                quiet=False,
                installer="pip",
            ),
        ]
        mock_installer.assert_has_calls(installer_calls)


class TestGetSitePackagesLocation:
    def test_returns_sysconfig_purelib(self):
        result = ValidatorPackageService.get_site_packages_location()
        import sysconfig

        assert result == sysconfig.get_paths()["purelib"]


class TestDetectInstaller:
    @pytest.mark.parametrize(
        ("env_value", "uv_available", "expected"),
        [
            ("uv", True, "uv"),
            ("pip", True, "pip"),
            ("uv", False, "uv"),
            ("", True, "uv"),
            ("", False, "pip"),
        ],
        ids=[
            "env_uv_with_uv_available",
            "env_pip_overrides_uv",
            "env_uv_without_uv_binary",
            "auto_detect_uv",
            "fallback_to_pip",
        ],
    )
    def test_detect_installer(self, env_value, uv_available, expected):
        with (
            patch.dict(os.environ, {_GUARDRAILS_INSTALLER_ENV: env_value}),
            patch(
                "guardrails.hub.validator_package_service.shutil.which",
                return_value="/usr/bin/uv" if uv_available else None,
            ),
        ):
            assert ValidatorPackageService.detect_installer() == expected


class TestGetRegistryPath:
    def test_returns_project_level_path(self, mocker):
        mocker.patch(
            "guardrails.hub.validator_package_service.os.getcwd",
            return_value="/my/project",
        )
        result = ValidatorPackageService.get_registry_path()
        assert result == Path("/my/project/.guardrails/hub_registry.json")


class TestRegisterValidator:
    def test_creates_new_registry(self, tmp_path, mocker):
        mocker.patch.object(
            ValidatorPackageService,
            "get_registry_path",
            return_value=tmp_path / ".guardrails" / "hub_registry.json",
        )
        manifest = Manifest.from_dict(
            {
                "id": "guardrails/detect_pii",
                "name": "detect_pii",
                "author": {"name": "test", "email": "t@t.com"},
                "maintainers": [],
                "repository": {"url": "https://github.com/test/test"},
                "namespace": "guardrails",
                "packageName": "detect-pii",
                "moduleName": "detect_pii",
                "description": "Detect PII",
                "exports": ["DetectPII"],
                "tags": {},
            }
        )
        manifest = cast(Manifest, manifest)

        ValidatorPackageService.register_validator(manifest)

        registry_file = tmp_path / ".guardrails" / "hub_registry.json"
        assert registry_file.exists()
        registry = json.loads(registry_file.read_text())
        assert registry["version"] == 1
        assert "guardrails/detect_pii" in registry["validators"]
        entry = registry["validators"]["guardrails/detect_pii"]
        assert entry["import_path"] == "guardrails_grhub_detect_pii"
        assert entry["exports"] == ["DetectPII"]
        assert entry["package_name"] == "guardrails-grhub-detect-pii"
        assert "installed_at" in entry

    def test_appends_to_existing_registry(self, tmp_path, mocker):
        registry_dir = tmp_path / ".guardrails"
        registry_dir.mkdir()
        registry_file = registry_dir / "hub_registry.json"
        existing = {
            "version": 1,
            "validators": {
                "guardrails/regex_match": {
                    "import_path": "guardrails_grhub_regex_match",
                    "exports": ["RegexMatch"],
                    "installed_at": "2025-01-01T00:00:00+00:00",
                    "package_name": "guardrails-grhub-regex-match",
                }
            },
        }
        registry_file.write_text(json.dumps(existing))

        mocker.patch.object(
            ValidatorPackageService,
            "get_registry_path",
            return_value=registry_file,
        )
        manifest = Manifest.from_dict(
            {
                "id": "guardrails/detect_pii",
                "name": "detect_pii",
                "author": {"name": "test", "email": "t@t.com"},
                "maintainers": [],
                "repository": {"url": "https://github.com/test/test"},
                "namespace": "guardrails",
                "packageName": "detect-pii",
                "moduleName": "detect_pii",
                "description": "Detect PII",
                "exports": ["DetectPII"],
                "tags": {},
            }
        )
        manifest = cast(Manifest, manifest)

        ValidatorPackageService.register_validator(manifest)

        registry = json.loads(registry_file.read_text())
        assert len(registry["validators"]) == 2
        assert "guardrails/regex_match" in registry["validators"]
        assert "guardrails/detect_pii" in registry["validators"]

    def test_overwrites_existing_entry(self, tmp_path, mocker):
        registry_dir = tmp_path / ".guardrails"
        registry_dir.mkdir()
        registry_file = registry_dir / "hub_registry.json"
        existing = {
            "version": 1,
            "validators": {
                "guardrails/detect_pii": {
                    "import_path": "guardrails_grhub_detect_pii",
                    "exports": ["DetectPII"],
                    "installed_at": "2025-01-01T00:00:00+00:00",
                    "package_name": "guardrails-grhub-detect-pii",
                }
            },
        }
        registry_file.write_text(json.dumps(existing))

        mocker.patch.object(
            ValidatorPackageService,
            "get_registry_path",
            return_value=registry_file,
        )
        manifest = Manifest.from_dict(
            {
                "id": "guardrails/detect_pii",
                "name": "detect_pii",
                "author": {"name": "test", "email": "t@t.com"},
                "maintainers": [],
                "repository": {"url": "https://github.com/test/test"},
                "namespace": "guardrails",
                "packageName": "detect-pii",
                "moduleName": "detect_pii",
                "description": "Detect PII",
                "exports": ["DetectPII", "DetectPIIv2"],
                "tags": {},
            }
        )
        manifest = cast(Manifest, manifest)

        ValidatorPackageService.register_validator(manifest)

        registry = json.loads(registry_file.read_text())
        assert len(registry["validators"]) == 1
        assert registry["validators"]["guardrails/detect_pii"]["exports"] == [
            "DetectPII",
            "DetectPIIv2",
        ]


class TestInstallHubModuleWithInstaller:
    @pytest.mark.parametrize(
        ("detected_installer",),
        [("uv",), ("pip",)],
        ids=["uv_installer", "pip_installer"],
    )
    def test_uses_detected_installer(self, mocker, detected_installer):
        mock_installer = mocker.patch(
            "guardrails.hub.validator_package_service.installer_process",
            return_value="Success",
        )
        mock_settings = mocker.patch(
            "guardrails.hub.validator_package_service.settings"
        )
        mock_settings.rc.token = "mock-token"
        mocker.patch.object(
            ValidatorPackageService,
            "detect_installer",
            return_value=detected_installer,
        )

        ValidatorPackageService.install_hub_module("guardrails/detect_pii")

        mock_installer.assert_called_once_with(
            "install",
            "guardrails-grhub-detect-pii[validators]",
            [
                "--index-url=https://__token__:mock-token@pypi.guardrailsai.com/simple",
                "--extra-index-url=https://pypi.org/simple",
            ],
            quiet=False,
            installer=detected_installer,
        )
