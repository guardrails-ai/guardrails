from unittest.mock import call

import pytest

from guardrails.cli.server.module_manifest import ModuleManifest
from tests.unit_tests.mocks.mock_file import MockFile


class TestInstall:
    def test_exits_early_if_uri_is_not_valid(self, mocker):
        mock_logger_error = mocker.patch("guardrails.cli.hub.install.logger.error")

        from guardrails.cli.hub.install import install, sys

        sys_exit_spy = mocker.spy(sys, "exit")

        with pytest.raises(SystemExit):
            install("not a hub uri")

        mock_logger_error.assert_called_once_with("Invalid URI!")
        sys_exit_spy.assert_called_once_with(1)

    def test_happy_path(self, mocker):
        mock_logger_log = mocker.patch("guardrails.cli.hub.install.logger.log")

        mock_get_validator_manifest = mocker.patch(
            "guardrails.cli.hub.install.get_validator_manifest"
        )
        manifest = ModuleManifest.from_dict(
            {
                "id": "id",
                "name": "name",
                "author": {"name": "me", "email": "me@me.me"},
                "maintainers": [],
                "repository": {"url": "some-repo"},
                "namespace": "guardrails",
                "package_name": "test-validator",
                "module_name": "test_validator",
                "exports": ["TestValidator"],
                "tags": {},
            }
        )
        mock_get_validator_manifest.return_value = manifest

        mock_get_site_packages_location = mocker.patch(
            "guardrails.cli.hub.install.get_site_packages_location"
        )
        site_packages = "./.venv/lib/python3.X/site-packages"
        mock_get_site_packages_location.return_value = site_packages

        mock_install_hub_module = mocker.patch(
            "guardrails.cli.hub.install.install_hub_module"
        )
        mock_run_post_install = mocker.patch(
            "guardrails.cli.hub.install.run_post_install"
        )
        mock_add_to_hub_init = mocker.patch(
            "guardrails.cli.hub.install.add_to_hub_inits"
        )

        from guardrails.cli.hub.install import install

        install("hub://guardrails/test-validator")

        log_calls = [
            call(level=5, msg="Installing hub://guardrails/test-validator..."),
            call(
                level=5,
                msg="✅Successfully installed hub://guardrails/test-validator!\n\nImport validator:\nfrom guardrails.hub import TestValidator\n\nGet more info:\nhttps://hub.guardrailsai.com/validator/id\n",  # noqa
            ),  # noqa
        ]
        assert mock_logger_log.call_count == 2
        mock_logger_log.assert_has_calls(log_calls)

        mock_get_validator_manifest.assert_called_once_with("guardrails/test-validator")

        assert mock_get_site_packages_location.call_count == 1

        mock_install_hub_module.assert_called_once_with(manifest, site_packages)

        mock_run_post_install.assert_called_once_with(manifest, site_packages)

        mock_add_to_hub_init.assert_called_once_with(manifest, site_packages)


class TestPipProcess:
    def test_no_package_string_format(self, mocker):
        mock_logger_debug = mocker.patch("guardrails.cli.hub.install.logger.debug")

        mock_sys_executable = mocker.patch("guardrails.cli.hub.install.sys.executable")

        mock_subprocess_check_output = mocker.patch(
            "guardrails.cli.hub.install.subprocess.check_output"
        )
        mock_subprocess_check_output.return_value = str.encode("string output")

        from guardrails.cli.hub.install import pip_process

        response = pip_process("inspect", flags=["--path=./install-here"])

        assert mock_logger_debug.call_count == 2
        debug_calls = [
            call("running pip inspect --path=./install-here "),
            call("decoding output from pip inspect "),
        ]
        mock_logger_debug.assert_has_calls(debug_calls)

        mock_subprocess_check_output.assert_called_once_with(
            [mock_sys_executable, "-m", "pip", "inspect", "--path=./install-here"]
        )

        assert response == "string output"

    def test_json_format(self, mocker):
        mock_logger_debug = mocker.patch("guardrails.cli.hub.install.logger.debug")

        mock_sys_executable = mocker.patch("guardrails.cli.hub.install.sys.executable")

        mock_subprocess_check_output = mocker.patch(
            "guardrails.cli.hub.install.subprocess.check_output"
        )
        mock_subprocess_check_output.return_value = str.encode("json output")

        class MockBytesHeaderParser:
            def parsebytes(self, *args):
                return {"output": "json"}

        mock_bytes_parser = mocker.patch("guardrails.cli.hub.install.BytesHeaderParser")
        mock_bytes_header_parser = MockBytesHeaderParser()
        mock_bytes_parser.return_value = mock_bytes_header_parser

        from guardrails.cli.hub.install import pip_process

        response = pip_process("show", "pip", format="json")

        assert mock_logger_debug.call_count == 3
        debug_calls = [
            call("running pip show  pip"),
            call("decoding output from pip show pip"),
            call(
                "json parse exception in decoding output from pip show pip. Falling back to accumulating the byte stream"  # noqa
            ),
        ]
        mock_logger_debug.assert_has_calls(debug_calls)

        mock_subprocess_check_output.assert_called_once_with(
            [mock_sys_executable, "-m", "pip", "show", "pip"]
        )

        assert response == {"output": "json"}

    def test_called_process_error(self, mocker):
        mock_logger_error = mocker.patch("guardrails.cli.hub.install.logger.error")
        mock_logger_debug = mocker.patch("guardrails.cli.hub.install.logger.debug")
        mock_sys_executable = mocker.patch("guardrails.cli.hub.install.sys.executable")
        mock_subprocess_check_output = mocker.patch(
            "guardrails.cli.hub.install.subprocess.check_output"
        )

        from subprocess import CalledProcessError

        mock_subprocess_check_output.side_effect = CalledProcessError(1, "something")

        from guardrails.cli.hub.install import pip_process, sys

        sys_exit_spy = mocker.spy(sys, "exit")

        with pytest.raises(SystemExit):
            pip_process("inspect")

            mock_logger_debug.assert_called_once_with("running pip inspect  ")

            mock_subprocess_check_output.assert_called_once_with(
                [mock_sys_executable, "-m", "pip", "inspect"]
            )

            mock_logger_error.assert_called_once_with(
                "Failed to inspect \nExit code: 1\nstdout: "
            )

            sys_exit_spy.assert_called_once_with(1)

    def test_other_exception(self, mocker):
        error = ValueError("something went wrong")
        mock_logger_debug = mocker.patch("guardrails.cli.hub.install.logger.debug")
        mock_logger_debug.side_effect = error

        mock_logger_error = mocker.patch("guardrails.cli.hub.install.logger.error")

        from guardrails.cli.hub.install import pip_process, sys

        sys_exit_spy = mocker.spy(sys, "exit")

        with pytest.raises(SystemExit):
            pip_process("inspect")

            mock_logger_debug.assert_called_once_with("running pip inspect  ")

            mock_logger_error.assert_called_once_with(
                "An unexpected exception occurred while try to inspect !", error
            )

            sys_exit_spy.assert_called_once_with(1)


def test_get_site_packages_location(mocker):
    mock_pip_process = mocker.patch("guardrails.cli.hub.install.pip_process")
    mock_pip_process.return_value = {"Location": "/site-pacakges"}

    from guardrails.cli.hub.install import get_site_packages_location

    response = get_site_packages_location()

    mock_pip_process.assert_called_once_with("show", "pip", format="json")

    assert response == "/site-pacakges"


@pytest.mark.parametrize(
    "manifest,expected",
    [
        (
            ModuleManifest.from_dict(
                {
                    "id": "id",
                    "name": "name",
                    "author": {"name": "me", "email": "me@me.me"},
                    "maintainers": [],
                    "repository": {"url": "some-repo"},
                    "namespace": "guardrails-ai",
                    "package_name": "test-validator",
                    "module_name": "test_validator",
                    "exports": ["TestValidator"],
                    "tags": {},
                }
            ),
            ["guardrails_ai", "test_validator"],
        ),
        (
            ModuleManifest.from_dict(
                {
                    "id": "id",
                    "name": "name",
                    "author": {"name": "me", "email": "me@me.me"},
                    "maintainers": [],
                    "repository": {"url": "some-repo"},
                    "namespace": "",
                    "package_name": "test-validator",
                    "module_name": "test_validator",
                    "exports": ["TestValidator"],
                    "tags": {},
                }
            ),
            ["test_validator"],
        ),
    ],
)
def test_get_org_and_package_dirs(manifest, expected):
    from guardrails.cli.hub.install import get_org_and_package_dirs

    actual = get_org_and_package_dirs(manifest)

    assert actual == expected


def test_get_hub_directory():
    manifest = ModuleManifest.from_dict(
        {
            "id": "id",
            "name": "name",
            "author": {"name": "me", "email": "me@me.me"},
            "maintainers": [],
            "repository": {"url": "some-repo"},
            "namespace": "guardrails-ai",
            "package_name": "test-validator",
            "module_name": "test_validator",
            "exports": ["TestValidator"],
            "tags": {},
        }
    )

    from guardrails.cli.hub.install import get_hub_directory

    hub_dir = get_hub_directory(manifest, "./site-packages")

    assert hub_dir == "./site-packages/guardrails/hub/guardrails_ai/test_validator"


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
        mock_open = mocker.patch("guardrails.cli.hub.install.open")
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

        mock_is_file = mocker.patch("guardrails.cli.hub.install.os.path.isfile")
        mock_is_file.return_value = True

        from guardrails.cli.hub.install import add_to_hub_inits

        add_to_hub_inits(manifest, site_packages)

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
        mock_open = mocker.patch("guardrails.cli.hub.install.open")
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

        mock_is_file = mocker.patch("guardrails.cli.hub.install.os.path.isfile")
        mock_is_file.return_value = True

        from guardrails.cli.hub.install import add_to_hub_inits

        add_to_hub_inits(manifest, site_packages)

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
        mock_open = mocker.patch("guardrails.cli.hub.install.open")
        mock_open.side_effect = [hub_init_file, ns_init_file]

        mock_hub_read = mocker.patch.object(hub_init_file, "read")
        mock_hub_read.return_value = "from guardrails.hub.guardrails_ai.test_validator.validator import TestValidator"  # noqa

        mock_ns_read = mocker.patch.object(ns_init_file, "read")
        mock_ns_read.return_value = ""

        ns_seek_spy = mocker.spy(ns_init_file, "seek")
        ns_write_spy = mocker.spy(ns_init_file, "write")
        ns_close_spy = mocker.spy(ns_init_file, "close")

        mock_is_file = mocker.patch("guardrails.cli.hub.install.os.path.isfile")
        mock_is_file.return_value = False

        from guardrails.cli.hub.install import add_to_hub_inits

        add_to_hub_inits(manifest, site_packages)

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
            "guardrails.cli.hub.install.subprocess.check_output"
        )
        from guardrails.cli.hub.install import run_post_install

        run_post_install(manifest, "./site_packages")

        assert mock_subprocess_check_output.call_count == 0

    def test_runs_script_if_exists(self, mocker):
        mock_subprocess_check_output = mocker.patch(
            "guardrails.cli.hub.install.subprocess.check_output"
        )
        mock_sys_executable = mocker.patch("guardrails.cli.hub.install.sys.executable")
        mock_isfile = mocker.patch("guardrails.cli.hub.install.os.path.isfile")
        mock_isfile.return_value = True
        from guardrails.cli.hub.install import run_post_install

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

        run_post_install(manifest, "./site_packages")

        assert mock_subprocess_check_output.call_count == 1
        mock_subprocess_check_output.assert_called_once_with(
            [
                mock_sys_executable,
                "./site_packages/guardrails/hub/guardrails_ai/test_validator/validator/post_install.py",  # noqa
            ]
        )


@pytest.mark.parametrize(
    "manifest,expected",
    [
        (
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
            "git+some-repo",
        ),
        (
            ModuleManifest.from_dict(
                {
                    "id": "id",
                    "name": "name",
                    "author": {"name": "me", "email": "me@me.me"},
                    "maintainers": [],
                    "repository": {"url": "git+some-repo"},
                    "namespace": "guardrails-ai",
                    "package_name": "test-validator",
                    "module_name": "validator",
                    "exports": ["TestValidator"],
                    "tags": {},
                    "post_install": "",
                }
            ),
            "git+some-repo",
        ),
        (
            ModuleManifest.from_dict(
                {
                    "id": "id",
                    "name": "name",
                    "author": {"name": "me", "email": "me@me.me"},
                    "maintainers": [],
                    "repository": {"url": "git+some-repo", "branch": "prod"},
                    "namespace": "guardrails-ai",
                    "package_name": "test-validator",
                    "module_name": "validator",
                    "exports": ["TestValidator"],
                    "tags": {},
                    "post_install": "",
                }
            ),
            "git+some-repo@prod",
        ),
    ],
)
def test_get_install_url(manifest, expected):
    from guardrails.cli.hub.install import get_install_url

    actual = get_install_url(manifest)

    assert actual == expected


def test_install_hub_module(mocker):
    mock_get_install_url = mocker.patch("guardrails.cli.hub.install.get_install_url")
    mock_get_install_url.return_value = "mock-install-url"

    mock_get_hub_directory = mocker.patch(
        "guardrails.cli.hub.install.get_hub_directory"
    )
    mock_get_hub_directory.return_value = "mock/install/directory"

    mock_pip_process = mocker.patch("guardrails.cli.hub.install.pip_process")
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

    from guardrails.cli.hub.install import install_hub_module

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
    install_hub_module(manifest, site_packages)

    mock_get_install_url.assert_called_once_with(manifest)
    mock_get_hub_directory.assert_called_once_with(manifest, site_packages)

    assert mock_pip_process.call_count == 5
    pip_calls = [
        call(
            "install",
            "mock-install-url",
            ["--target=mock/install/directory", "--no-deps", "-q"],
        ),
        call("inspect", flags=["--path=mock/install/directory"], format="json"),
        call("install", "rstr"),
        call("install", "openai<2"),
        call("install", "pydash>=7.0.6,<8.0.0"),
    ]
    mock_pip_process.assert_has_calls(pip_calls)
