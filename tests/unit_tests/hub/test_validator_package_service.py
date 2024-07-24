from pathlib import Path
import pytest
import sys
from unittest.mock import call, patch, MagicMock

from guardrails.cli.server.module_manifest import ModuleManifest
from guardrails.hub.validator_package_service import (
    ValidatorPackageService,
    InvalidHubInstallURL,
)
from tests.unit_tests.mocks.mock_file import MockFile


class TestAddToHubInits:
    def test_closes_early_if_already_added(self, mocker):
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
        mock_module = MagicMock()
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

    # --

    # @patch('guardrails.hub.validator_package_service.importlib')
    # def test_reload_module_already_imported(self, mock_importlib):
    #     # Setup
    #     sys.modules["guardrails.hub"] = MagicMock()
    #     mock_module = MagicMock()
    #     mock_importlib.reload.return_value = mock_module

    #     # Test
    #     reloaded_module = ValidatorPackageService.reload_module("guardrails.hub.guardrails.contains_string.validator")

    #     # Assert
    #     assert reloaded_module == mock_module
    #     mock_importlib.reload.assert_called_once_with(sys.modules["guardrails.hub"])

    # @patch('guardrails.hub.validator_package_service.importlib')
    # def test_reload_module_not_imported(self, mock_importlib):
    #     # Setup
    #     mock_module = MagicMock()
    #     mock_importlib.import_module.return_value = mock_module

    #     # Test
    #     reloaded_module = ValidatorPackageService.reload_module("guardrails.hub")

    #     # Assert
    #     assert reloaded_module == mock_module
    #     mock_importlib.import_module.assert_called_once_with("guardrails.hub")

    # @patch('guardrails.hub.validator_package_service.importlib')
    # def test_reload_module_module_not_found(self, mock_importlib):
    #     # Setup
    #     mock_importlib.import_module.side_effect = ModuleNotFoundError("Module not found")

    #     # Test
    #     reloaded_module = ValidatorPackageService.reload_module("guardrails.hub")

    #     # Assert
    #     assert reloaded_module is None
    #     mock_importlib.import_module.assert_called_once_with("guardrails.hub")

    # @patch('guardrails.hub.validator_package_service.importlib')
    # def test_reload_module_unexpected_exception(self, mock_importlib):
    #     # Setup
    #     mock_importlib.import_module.side_effect = Exception("Unexpected exception")

    #     # Test
    #     reloaded_module = ValidatorPackageService.reload_module("guardrails.hub")

    #     # Assert
    #     assert reloaded_module is None
    #     mock_importlib.import_module.assert_called_once_with("guardrails.hub")


class TestRunPostInstall:
    @pytest.mark.parametrize(
        "manifest",
        [
            ModuleManifest.from_dict(
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
            ),
            ModuleManifest.from_dict(
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
        self.manifest = ModuleManifest(
            id="test-id",
            name="test-validator",
            namespace="guardrails-ai",
            package_name="test-package",
            module_name="test_module",
            exports=["TestValidator"],
            repository={"url": "https://github.com/example/test", "branch": "main"},
            post_install="post_install_script.py",
            author="Alejandro",
            maintainers=["Alejandro"],
            encoder="json",
        )
        self.site_packages = "/fake/site-packages"

    @patch(
        "guardrails.hub.validator_package_service.ValidatorPackageService.get_module_name"
    )
    @patch(
        "guardrails.hub.validator_package_service.ValidatorPackageService.install__prep"
    )
    @patch(
        "guardrails.hub.validator_package_service.ValidatorPackageService.install__pip_install_hub_module"
    )
    @patch(
        "guardrails.hub.validator_package_service.ValidatorPackageService.install__post_install"
    )
    def test_install(
        self, mock_post_install, mock_install_module, mock_prep, mock_get_module_name
    ):
        # Setup
        mock_get_module_name.return_value = "test-module"
        mock_prep.return_value = (self.manifest, self.site_packages)
        expected_validator = MagicMock()

        with patch(
            "guardrails.hub.validator_package_service.ValidatorPackageService.get_validator_from_manifest",
            return_value=expected_validator,
        ):
            # Test
            validator = ValidatorPackageService.install("hub://test-module")

            # Assert
            assert validator == expected_validator
            mock_get_module_name.assert_called_once_with("hub://test-module")
            mock_prep.assert_called_once_with("test-module")
            mock_install_module.assert_called_once_with(
                self.manifest, self.site_packages, quiet=True
            )
            mock_post_install.assert_called_once_with(self.manifest, self.site_packages)

    @patch("guardrails.hub.validator_package_service.get_validator_manifest")
    @patch(
        "guardrails.hub.validator_package_service.ValidatorPackageService.get_site_packages_location"
    )
    def test_install__prep(
        self, mock_get_site_packages_location, mock_get_validator_manifest
    ):
        # Setup
        mock_get_validator_manifest.return_value = self.manifest
        mock_get_site_packages_location.return_value = self.site_packages

        # Test
        manifest, site_packages = ValidatorPackageService.install__prep("test-module")

        # Assert
        assert manifest == self.manifest
        assert site_packages == self.site_packages
        mock_get_validator_manifest.assert_called_once_with("test-module")
        mock_get_site_packages_location.assert_called_once()

    @patch(
        "guardrails.hub.validator_package_service.ValidatorPackageService.run_post_install"
    )
    @patch(
        "guardrails.hub.validator_package_service.ValidatorPackageService.add_to_hub_inits"
    )
    def test_install__post_install(self, mock_add_to_hub_inits, mock_run_post_install):
        ValidatorPackageService.install__post_install(self.manifest, self.site_packages)
        mock_add_to_hub_inits.assert_called_once_with(self.manifest, self.site_packages)
        mock_run_post_install.assert_called_once_with(self.manifest, self.site_packages)

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

    def test_get_org_and_package_dirs(self):
        org_package_dirs = ValidatorPackageService.get_org_and_package_dirs(
            self.manifest
        )
        assert org_package_dirs == ["guardrails_ai", "test_package"]

    def test_get_module_name_valid(self):
        # Test
        module_name = ValidatorPackageService.get_module_name("hub://test-module")

        # Assert
        assert module_name == "test-module"

    def test_get_module_name_invalid(self):
        # Test & Assert
        with pytest.raises(InvalidHubInstallURL):
            ValidatorPackageService.get_module_name("invalid-uri")


# import pytest
# from unittest.mock import patch, MagicMock
# from guardrails.cli.server.module_manifest import ModuleManifest
# from guardrails.hub.validator_package_service import (
#     ValidatorPackageService,
#     FailedPackageInstallationPostInstall,
#     InvalidHubInstallURL,
#     FailedPackageInspection,
# )

# class TestValidatorPackageService:
#     def setup_method(self):
#         self.manifest = ModuleManifest(
#             id="test-id",
#             name="test-validator",
#             namespace="guardrails-ai",
#             package_name="test-package",
#             module_name="test_module",
#             exports=["TestValidator"],
#             repository={"url": "https://github.com/example/test", "branch": "main"},
#             post_install="post_install_script.py",
#             author="Alejandro",
#             maintainers=["Alejandro"],
#             encoder="json",
#         )
#         self.site_packages = "/fake/site-packages"

#     @patch('guardrails.hub.validator_package_service.ValidatorPackageService.get_module_name')
#     @patch('guardrails.hub.validator_package_service.ValidatorPackageService.install__prep')
#     @patch('guardrails.hub.validator_package_service.ValidatorPackageService.install__pip_install_hub_module')
#     @patch('guardrails.hub.validator_package_service.ValidatorPackageService.install__post_install')
#     def test_install(self, mock_post_install, mock_install_module, mock_prep, mock_get_module_name):
#         # Setup
#         mock_get_module_name.return_value = "test-module"
#         mock_prep.return_value = (self.manifest, self.site_packages)
#         expected_validator = MagicMock()

#         with patch('guardrails.hub.validator_package_service.ValidatorPackageService.get_validator_from_manifest', return_value=expected_validator):
#             # Test
#             validator = ValidatorPackageService.install('hub://test-module')

#             # Assert
#             assert validator == expected_validator
#             mock_get_module_name.assert_called_once_with('hub://test-module')
#             mock_prep.assert_called_once_with("test-module")
#             mock_install_module.assert_called_once_with(self.manifest, self.site_packages, quiet=True)
#             mock_post_install.assert_called_once_with(self.manifest, self.site_packages)

#     @patch('guardrails.hub.validator_package_service.get_validator_manifest')
#     @patch('guardrails.hub.validator_package_service.ValidatorPackageService.get_site_packages_location')
#     def test_install__prep(self, mock_get_site_packages_location, mock_get_validator_manifest):
#         # Setup
#         mock_get_validator_manifest.return_value = self.manifest
#         mock_get_site_packages_location.return_value = self.site_packages

#         # Test
#         manifest, site_packages = ValidatorPackageService.install__prep("test-module")

#         # Assert
#         assert manifest == self.manifest
#         assert site_packages == self.site_packages
#         mock_get_validator_manifest.assert_called_once_with("test-module")
#         mock_get_site_packages_location.assert_called_once()


#     def test_get_module_name_valid(self):
#         # Test
#         module_name = ValidatorPackageService.get_module_name("hub://test-module")

#         # Assert
#         assert module_name == "test-module"

#     def test_get_module_name_invalid(self):
#         # Test & Assert
#         with pytest.raises(InvalidHubInstallURL):
#             ValidatorPackageService.get_module_name("invalid-uri")

#     def test_get_site_packages_location(self):
#         # Test
#         site_packages_path = ValidatorPackageService.get_site_packages_location()

#         # Assert
#         assert site_packages_path == "/fake/site-packages"

#

#     def test_get_validator_from_manifest(self):
#         # Setup
#         expected_validator = MagicMock()
#         mock_reload_module = MagicMock()
#         mock_reload_module.return_value = expected_validator
#         ValidatorPackageService.reload_module = mock_reload_module

#         # Test
#         validator = ValidatorPackageService.get_validator_from_manifest(self.manifest)

#         # Assert
#         assert validator == expected_validator
#         mock_reload_module.assert_called_once_with("guardrails.hub.guardrails_ai.test_module")

#     def test_get_org_and_package_dirs(self):
#         # Test
#         org_package_dirs = ValidatorPackageService.get_org_and_package_dirs(self.manifest)

#         # Assert
#         assert org_package_dirs == ["guardrails_ai", "test_package"]

#     def test_add_to_hub_inits_existing_hub_init(self):
#         # Setup
#         hub_init_content = "import guardrails.hub.test_module"
#         with patch('builtins.open', create=True) as mock_open:
#             mock_open.return_value.__enter__.return_value.read.return_value = hub_init_content

#             # Test
#             ValidatorPackageService.add_to_hub_inits(self.manifest, self.site_packages)

#             # Assert
#             mock_open.assert_called_once_with("/fake/site-packages/guardrails/hub/__init__.py", "a+")
#             mock_open.return_value.__enter__.return_value.write.assert_not_called()

#     def test_add_to_hub_inits_new_hub_init(self):
#         # Setup
#         with patch('builtins.open', create=True) as mock_open:
#             mock_open.return_value.__enter__.return_value.read.return_value = ""

#             # Test
#             ValidatorPackageService.add_to_hub_inits(self.manifest, self.site_packages)

#             # Assert
#             mock_open.assert_called_once_with("/fake/site-packages/guardrails/hub/__init__.py", "a+")
#             mock_open.return_value.__enter__.return_value.write.assert_called_once_with("from guardrails.hub.guardrails_ai.test_module import TestValidator")

#     def test_add_to_hub_inits_existing_namespace_init(self):
#         # Setup
#         namespace_init_content = "import guardrails.hub.guardrails_ai.test_module"
#         with patch('builtins.open', create=True) as mock_open:
#             mock_open.return_value.__enter__.return_value.read.return_value = namespace_init_content

#             # Test
#             ValidatorPackageService.add_to_hub_inits(self.manifest, self.site_packages)

#             # Assert
#             mock_open.assert_called_once_with("/fake/site-packages/guardrails/hub/guardrails_ai/__init__.py", "a+")
#             mock_open.return_value.__enter__.return_value.write.assert_not_called()

#     def test_add_to_hub_inits_new_namespace_init(self):
#         # Setup
#         with patch('builtins.open', create=True) as mock_open:
#             mock_open.return_value.__enter__.return_value.read.return_value = ""

#             # Test
#             ValidatorPackageService.add_to_hub_inits(self.manifest, self.site_packages)

#             # Assert
#             mock_open.assert_called_once_with("/fake/site-packages/guardrails/hub/guardrails_ai/__init__.py", "w")
#             mock_open.return_value.__enter__.return_value.write.assert_called_once_with("from guardrails.hub.guardrails_ai.test_module import TestValidator")

#     def test_get_module_path_package(self):
#         # Setup
#         sys.modules["pip"] = MagicMock()
#         sys.modules["pip"].__path__ = ["/fake/site-packages/pip"]

#         # Test
#         module_path = ValidatorPackageService.get_module_path("pip")

#         # Assert
#         assert module_path == "/fake/site-packages/pip"

#     def test_get_module_path_module(self):
#         # Setup
#         sys.modules["pip"] = MagicMock()
#         sys.modules["pip"].__path__ = None

#         # Test
#         module_path = ValidatorPackageService.get_module_path("pip")

#         # Assert
#         assert module_path == "/fake/site-packages/pip"

#     def test_get_module_path_failed_to_locate_module(self):
#         # Setup
#         sys.modules["pip"] = MagicMock()
#         sys.modules["pip"].__path__ = None

#         # Test & Assert
#         with pytest.raises(FailedToLocateModule):
#             ValidatorPackageService.get_module_path("invalid-module")

#     def test_get_module_name(self):
#         # Test
#         module_name = ValidatorPackageService.get_module_name("hub://test-module")

#         # Assert
#         assert module_name == "test-module"

#     def test_get_module_name_invalid_uri(self):
#         # Test & Assert
#         with pytest.raises(InvalidHubInstallURL):
#             ValidatorPackageService.get_module_name("invalid-uri")

#     @patch('guardrails.hub.validator_package_service.subprocess')
#     def test_run_post_install_existing_script(self, mock_subprocess):
#         # Setup
#         self.manifest.post_install = "post_install_script.py"
#         mock_subprocess.check_output.return_value = b""

#         # Test
#         ValidatorPackageService.run_post_install(self.manifest, self.site_packages)

#         # Assert
#         mock_subprocess.check_output.assert_called_once_with([sys.executable, "/fake/site-packages/guardrails/hub/guardrails_ai/test_module/post_install_script.py"])

#     @patch('guardrails.hub.validator_package_service.subprocess')
#     def test_run_post_install_nonexistent_script(self, mock_subprocess):
#         # Setup
#         self.manifest.post_install = "nonexistent_script.py"

#         # Test
#         ValidatorPackageService.run_post_install(self.manifest, self.site_packages)

#         # Assert
#         mock_subprocess.check_output.assert_not_called()

#     @patch('guardrails.hub.validator_package_service.subprocess')
#     def test_run_post_install_failed_process_error(self, mock_subprocess):
#         # Setup
#         self.manifest.post_install = "post_install_script.py"
#         mock_subprocess.check_output.side_effect = subprocess.CalledProcessError(1, "command", output=b"Error")

#         # Test & Assert
#         with pytest.raises(FailedPackageInstallationPostInstall):
#             ValidatorPackageService.run_post_install(self.manifest, self.site_packages)

#     @patch('guardrails.hub.validator_package_service.subprocess')
#     def test_run_post_install_unexpected_exception(self, mock_subprocess):
#         # Setup
#         self.manifest.post_install = "post_install_script.py"
#         mock_subprocess.check_output.side_effect = Exception("Unexpected exception")

#         # Test & Assert
#         with pytest.raises(FailedPackageInstallationPostInstall):
#             ValidatorPackageService.run_post_install(self.manifest, self.site_packages)

#     def test_get_hub_directory(self):
#         # Test
#         hub_directory = ValidatorPackageService.get_hub_directory(self.manifest, self.site_packages)

#         # Assert
#         assert hub_directory == "/fake/site-packages/guardrails/hub/guardrails_ai/test_package"

#     @patch('guardrails.hub.validator_package_service.pip_process')
#     def test_install__pip_install_hub_module(self, mock_pip_process):
#         # Setup
#         mock_pip_process.side_effect = lambda *args, **kwargs: f"pip {args} {kwargs}"

#         # Test
#         ValidatorPackageService.install__pip_install_hub_module(self.manifest, self.site_packages, quiet=False)

#         # Assert
#         mock_pip_process.assert_called_once_with("install", "git+https://github.com/example/test@main", ["--target=/fake/site-packages/guardrails/hub/guardrails_ai/test_package", "--no-deps"], quiet=False)
