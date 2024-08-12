from pathlib import Path
import pytest
import sys
from unittest.mock import call, patch, MagicMock

from guardrails_hub_types import Manifest
from guardrails.hub.validator_package_service import (
    FailedToLocateModule,
    ValidatorPackageService,
    InvalidHubInstallURL,
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
                "id": "id",
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
        ns_init_file = MockFile()
        mock_open = mocker.patch("guardrails.hub.validator_package_service.open")
        mock_open.side_effect = [hub_init_file, ns_init_file]

        mock_hub_read = mocker.patch.object(hub_init_file, "read")
        mock_hub_read.return_value = "from guardrails.hub.guardrails_ai.test_validator.validator import helper, TestValidator"  # noqa

        hub_seek_spy = mocker.spy(hub_init_file, "seek")
        hub_write_spy = mocker.spy(hub_init_file, "write")
        hub_close_spy = mocker.spy(hub_init_file, "close")

        mock_ns_read = mocker.patch.object(ns_init_file, "read")
        mock_ns_read.return_value = "from guardrails.hub.guardrails_ai.test_validator.validator import helper, TestValidator"  # noqa

        ns_seek_spy = mocker.spy(ns_init_file, "seek")
        ns_write_spy = mocker.spy(ns_init_file, "write")
        ns_close_spy = mocker.spy(ns_init_file, "close")

        mock_is_file = mocker.patch(
            "guardrails.hub.validator_package_service.os.path.isfile"
        )
        mock_is_file.return_value = True

        from guardrails.hub.validator_package_service import ValidatorPackageService

        ValidatorPackageService.add_to_hub_inits(manifest, site_packages)

        assert mock_open.call_count == 2
        open_calls = [
            call("./site-packages/guardrails/hub/__init__.py", "a+"),
            call("./site-packages/guardrails/hub/guardrails_ai/__init__.py", "a+"),
        ]
        mock_open.assert_has_calls(open_calls)

        assert hub_seek_spy.call_count == 1
        assert mock_hub_read.call_count == 1
        assert hub_write_spy.call_count == 0
        assert hub_close_spy.call_count == 1

        mock_is_file.assert_called_once_with(
            "./site-packages/guardrails/hub/guardrails_ai/__init__.py"
        )
        assert ns_seek_spy.call_count == 1
        assert mock_ns_read.call_count == 1
        assert ns_write_spy.call_count == 0
        assert ns_close_spy.call_count == 1

    def test_appends_import_line_if_not_present(self, mocker):
        manifest = Manifest.from_dict(
            {
                "id": "id",
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
        ns_init_file = MockFile()
        mock_open = mocker.patch("guardrails.hub.validator_package_service.open")
        mock_open.side_effect = [hub_init_file, ns_init_file]

        mock_hub_read = mocker.patch.object(hub_init_file, "read")
        mock_hub_read.return_value = "from guardrails.hub.other_org.other_validator.validator import OtherValidator"  # noqa

        hub_seek_spy = mocker.spy(hub_init_file, "seek")
        hub_write_spy = mocker.spy(hub_init_file, "write")
        hub_close_spy = mocker.spy(hub_init_file, "close")

        mock_ns_read = mocker.patch.object(ns_init_file, "read")
        mock_ns_read.return_value = ""

        ns_seek_spy = mocker.spy(ns_init_file, "seek")
        ns_write_spy = mocker.spy(ns_init_file, "write")
        ns_close_spy = mocker.spy(ns_init_file, "close")

        mock_is_file = mocker.patch(
            "guardrails.hub.validator_package_service.os.path.isfile"
        )
        mock_is_file.return_value = True

        from guardrails.hub.validator_package_service import ValidatorPackageService

        ValidatorPackageService.add_to_hub_inits(manifest, site_packages)

        assert mock_open.call_count == 2
        open_calls = [
            call("./site-packages/guardrails/hub/__init__.py", "a+"),
            call("./site-packages/guardrails/hub/guardrails_ai/__init__.py", "a+"),
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
                "from guardrails.hub.guardrails_ai.test_validator.validator import TestValidator"  # noqa
            ),
        ]
        hub_write_spy.assert_has_calls(hub_write_calls)

        assert hub_close_spy.call_count == 1

        mock_is_file.assert_called_once_with(
            "./site-packages/guardrails/hub/guardrails_ai/__init__.py"
        )

        assert ns_seek_spy.call_count == 2
        ns_seek_calls = [call(0, 0), call(0, 2)]
        ns_seek_spy.assert_has_calls(ns_seek_calls)

        assert mock_ns_read.call_count == 1
        assert ns_write_spy.call_count == 1
        ns_write_spy.assert_called_once_with(
            "from guardrails.hub.guardrails_ai.test_validator.validator import TestValidator"  # noqa
        )
        assert ns_close_spy.call_count == 1

    def test_creates_namespace_init_if_not_exists(self, mocker):
        manifest = Manifest.from_dict(
            {
                "id": "id",
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
        ns_init_file = MockFile()
        mock_open = mocker.patch("guardrails.hub.validator_package_service.open")
        mock_open.side_effect = [hub_init_file, ns_init_file]

        mock_hub_read = mocker.patch.object(hub_init_file, "read")
        mock_hub_read.return_value = "from guardrails.hub.guardrails_ai.test_validator.validator import TestValidator"  # noqa

        mock_ns_read = mocker.patch.object(ns_init_file, "read")
        mock_ns_read.return_value = ""

        ns_seek_spy = mocker.spy(ns_init_file, "seek")
        ns_write_spy = mocker.spy(ns_init_file, "write")
        ns_close_spy = mocker.spy(ns_init_file, "close")

        mock_is_file = mocker.patch(
            "guardrails.hub.validator_package_service.os.path.isfile"
        )
        mock_is_file.return_value = False

        from guardrails.hub.validator_package_service import ValidatorPackageService

        ValidatorPackageService.add_to_hub_inits(manifest, site_packages)

        assert mock_open.call_count == 2
        open_calls = [
            call("./site-packages/guardrails/hub/__init__.py", "a+"),
            call("./site-packages/guardrails/hub/guardrails_ai/__init__.py", "w"),
        ]
        mock_open.assert_has_calls(open_calls)

        mock_is_file.assert_called_once_with(
            "./site-packages/guardrails/hub/guardrails_ai/__init__.py"
        )

        assert ns_seek_spy.call_count == 0
        assert mock_ns_read.call_count == 0
        assert ns_write_spy.call_count == 1
        ns_write_spy.assert_called_once_with(
            "from guardrails.hub.guardrails_ai.test_validator.validator import TestValidator"  # noqa
        )
        assert ns_close_spy.call_count == 1


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
                    "id": "id",
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
                    "id": "id",
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
                "id": "id",
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
                "post_install": "post_install.py",
            }
        )

        ValidatorPackageService.run_post_install(manifest, "./site_packages")

        assert mock_subprocess_check_output.call_count == 1
        mock_subprocess_check_output.assert_called_once_with(
            [
                mock_sys_executable,
                "./site_packages/guardrails/hub/guardrails_ai/test_validator/validator/post_install.py",  # noqa
            ]
        )


class TestValidatorPackageService:
    def setup_method(self):
        self.manifest = Manifest.from_dict(
            {
                "id": "id",
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

    @patch(
        "guardrails.hub.validator_package_service.ValidatorPackageService.get_module_path"
    )
    def test_get_site_packages_location(self, mock_get_module_path):
        mock_get_module_path.return_value = Path("/fake/site-packages/pip")
        site_packages_path = ValidatorPackageService.get_site_packages_location()
        assert site_packages_path == "/fake/site-packages"

    @patch(
        "guardrails.hub.validator_package_service.ValidatorPackageService.get_org_and_package_dirs"
    )
    @patch(
        "guardrails.hub.validator_package_service.ValidatorPackageService.reload_module"
    )
    def test_get_validator_from_manifest(
        self, mock_reload_module, mock_get_org_and_package_dirs
    ):
        mock_get_org_and_package_dirs.return_value = ["guardrails_ai", "test_package"]

        mock_validator_module = MagicMock()
        mock_reload_module.return_value = mock_validator_module

        ValidatorPackageService.get_validator_from_manifest(self.manifest)

        mock_reload_module.assert_called_once_with(
            f"guardrails.hub.guardrails_ai.test_package.{self.manifest.module_name}"
        )

    @pytest.mark.parametrize(
        "manifest,expected",
        [
            (
                Manifest.from_dict(
                    {
                        "id": "id",
                        "name": "name",
                        "author": {"name": "me", "email": "me@me.me"},
                        "maintainers": [],
                        "repository": {"url": "some-repo"},
                        "namespace": "guardrails-ai",
                        "packageName": "test-validator",
                        "moduleName": "test_validator",
                        "description": "description",
                        "exports": ["TestValidator"],
                        "tags": {},
                    }
                ),
                ["guardrails_ai", "test_validator"],
            ),
            (
                Manifest.from_dict(
                    {
                        "id": "id",
                        "name": "name",
                        "author": {"name": "me", "email": "me@me.me"},
                        "maintainers": [],
                        "repository": {"url": "some-repo"},
                        "namespace": "",
                        "packageName": "test-validator",
                        "moduleName": "test_validator",
                        "description": "description",
                        "exports": ["TestValidator"],
                        "tags": {},
                    }
                ),
                ["test_validator"],
            ),
        ],
    )
    def test_get_org_and_package_dirs(self, manifest, expected):
        from guardrails.hub.validator_package_service import ValidatorPackageService

        actual = ValidatorPackageService.get_org_and_package_dirs(manifest)
        assert actual == expected

    def test_get_module_name_valid(self):
        module_name = ValidatorPackageService.get_module_name("hub://test-module")
        assert module_name == "test-module"

    def test_get_module_name_invalid(self):
        with pytest.raises(InvalidHubInstallURL):
            ValidatorPackageService.get_module_name("invalid-uri")

    @pytest.mark.parametrize(
        "manifest,expected",
        [
            (
                Manifest.from_dict(
                    {
                        "id": "id",
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
                "git+some-repo",
            ),
            (
                Manifest.from_dict(
                    {
                        "id": "id",
                        "name": "name",
                        "author": {"name": "me", "email": "me@me.me"},
                        "maintainers": [],
                        "repository": {"url": "git+some-repo"},
                        "namespace": "guardrails-ai",
                        "packageName": "test-validator",
                        "moduleName": "validator",
                        "description": "description",
                        "exports": ["TestValidator"],
                        "tags": {},
                        "post_install": "",
                    }
                ),
                "git+some-repo",
            ),
            (
                Manifest.from_dict(
                    {
                        "id": "id",
                        "name": "name",
                        "author": {"name": "me", "email": "me@me.me"},
                        "maintainers": [],
                        "repository": {"url": "git+some-repo", "branch": "prod"},
                        "namespace": "guardrails-ai",
                        "packageName": "test-validator",
                        "moduleName": "validator",
                        "description": "description",
                        "exports": ["TestValidator"],
                        "tags": {},
                        "post_install": "",
                    }
                ),
                "git+some-repo@prod",
            ),
        ],
    )
    def test_get_install_url(self, manifest, expected):
        actual = ValidatorPackageService.get_install_url(manifest)
        assert actual == expected

    def test_get_hub_directory(self):
        hub_directory = ValidatorPackageService.get_hub_directory(
            self.manifest, self.site_packages
        )
        assert (
            hub_directory
            == "./.venv/lib/python3.X/site-packages/guardrails/hub/guardrails/test_validator"  # noqa
        )  # noqa

    def test_install_hub_module(self, mocker):
        mock_get_install_url = mocker.patch(
            "guardrails.hub.validator_package_service.ValidatorPackageService.get_install_url"
        )
        mock_get_install_url.return_value = "mock-install-url"

        mock_get_hub_directory = mocker.patch(
            "guardrails.hub.validator_package_service.ValidatorPackageService.get_hub_directory"
        )
        mock_get_hub_directory.return_value = "mock/install/directory"

        mock_pip_process = mocker.patch(
            "guardrails.hub.validator_package_service.pip_process"
        )
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

        manifest = Manifest.from_dict(
            {
                "id": "id",
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
        ValidatorPackageService.install_hub_module(manifest, site_packages)

        mock_get_install_url.assert_called_once_with(manifest)
        mock_get_hub_directory.assert_called_once_with(manifest, site_packages)

        assert mock_pip_process.call_count == 5
        pip_calls = [
            call(
                "install",
                "mock-install-url",
                ["--target=mock/install/directory", "--no-deps"],
                quiet=False,
            ),
            call(
                "inspect",
                flags=["--path=mock/install/directory"],
                format="json",
                quiet=False,
                no_color=True,
            ),
            call("install", "rstr", quiet=False),
            call("install", "openai<2", quiet=False),
            call("install", "pydash>=7.0.6,<8.0.0", quiet=False),
        ]
        mock_pip_process.assert_has_calls(pip_calls)
