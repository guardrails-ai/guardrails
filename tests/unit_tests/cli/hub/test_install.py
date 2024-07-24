from unittest.mock import call

import pytest
from typer.testing import CliRunner

from guardrails.cli.hub.install import hub_command, install
from guardrails.cli.server.module_manifest import ModuleManifest


class TestInstall:
    def test_exits_early_if_uri_is_not_valid(self, mocker):
        mock_logger_error = mocker.patch("guardrails.cli.hub.install.logger.error")

        from guardrails.cli.hub.install import install, sys

        sys_exit_spy = mocker.spy(sys, "exit")

        with pytest.raises(SystemExit):
            install("not a hub uri")

        mock_logger_error.assert_called_once_with("Invalid URI!")
        sys_exit_spy.assert_called_once_with(1)

    def test_install_local_models(self, mocker, monkeypatch):
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
                "tags": {"has_guardrails_endpoint": False},
            }
        )
        mock_get_validator_manifest.return_value = manifest

        mock_get_site_packages_location = mocker.patch(
            "guardrails.cli.hub.install.get_site_packages_location"
        )
        site_packages = "./.venv/lib/python3.X/site-packages"
        mock_get_site_packages_location.return_value = site_packages

        mocker.patch("guardrails.cli.hub.install.install_hub_module")

        mock_add_to_hub_init = mocker.patch(
            "guardrails.cli.hub.install.add_to_hub_inits"
        )

        monkeypatch.setattr("typer.confirm", lambda prompt, default=True: True)

        from guardrails.cli.hub.install import install

        install("hub://guardrails/test-validator", quiet=False)

        log_calls = [
            call(level=5, msg="Installing hub://guardrails/test-validator..."),
            call(
                level=5,
                msg="Skipping post install, models will not be downloaded for local "
                "inference.",
            ),
            call(
                level=5,
                msg="✅Successfully installed hub://guardrails/test-validator!\n\nImport validator:\nfrom guardrails.hub import TestValidator\n\nGet more info:\nhttps://hub.guardrailsai.com/validator/id\n",  # noqa
            ),  # noqa
        ]
        assert mock_logger_log.call_count == 3
        mock_logger_log.assert_has_calls(log_calls)

        mock_get_validator_manifest.assert_called_once_with("guardrails/test-validator")

        assert mock_get_site_packages_location.call_count == 1

        mock_add_to_hub_init.assert_called_once_with(manifest, site_packages)

    def test_happy_path(self, mocker, monkeypatch):
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
                "tags": {"has_guardrails_endpoint": True},
            }
        )
        mock_get_validator_manifest.return_value = manifest

        mock_get_site_packages_location = mocker.patch(
            "guardrails.cli.hub.install.get_site_packages_location"
        )
        site_packages = "./.venv/lib/python3.X/site-packages"
        mock_get_site_packages_location.return_value = site_packages

        mocker.patch("guardrails.cli.hub.install.install_hub_module")
        mocker.patch("guardrails.cli.hub.install.run_post_install")
        mocker.patch("guardrails.cli.hub.install.add_to_hub_inits")

        monkeypatch.setattr("typer.confirm", lambda _: False)

        install("hub://guardrails/test-validator", quiet=False)

        log_calls = [
            call(level=5, msg="Installing hub://guardrails/test-validator..."),
            call(
                level=5,
                msg="Skipping post install, models will not be downloaded for local inference.",  # noqa
            ),  # noqa
        ]

        assert mock_logger_log.call_count == 3
        mock_logger_log.assert_has_calls(log_calls)

        mock_get_validator_manifest.assert_called_once_with("guardrails/test-validator")

        assert mock_get_site_packages_location.call_count == 1

    def test_install_local_models_confirmation(self, mocker):
        # Mock dependencies
        mocker.patch("guardrails.cli.hub.install.get_site_packages_location")
        mocker.patch("guardrails.cli.hub.install.install_hub_module")
        mocker.patch("guardrails.cli.hub.install.run_post_install")
        mocker.patch("guardrails.cli.hub.install.add_to_hub_inits")

        # Create a manifest with Guardrails endpoint
        manifest_with_endpoint = ModuleManifest.from_dict(
            {
                "id": "test-id",
                "name": "test-name",
                "author": {"name": "test-author", "email": "test@email.com"},
                "maintainers": [],
                "repository": {"url": "test-repo"},
                "namespace": "test-namespace",
                "package_name": "test-package",
                "module_name": "test_module",
                "exports": ["TestValidator"],
                "tags": {"has_guardrails_endpoint": False},
            }
        )
        mocker.patch(
            "guardrails.cli.hub.install.get_validator_manifest",
            return_value=manifest_with_endpoint,
        )

        runner = CliRunner()

        # Run the install command with simulated user input
        result = runner.invoke(
            hub_command, ["install", "hub://test-namespace/test-package"]
        )

        # Check if the installation was successful
        assert result.exit_code == 0


class TestPipProcess:
    def test_no_package_string_format(self, mocker):
        mocker.patch("guardrails.cli.hub.install.os.environ", return_value={})
        mock_logger_debug = mocker.patch("guardrails.cli.hub.utils.logger.debug")

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
            [mock_sys_executable, "-m", "pip", "inspect", "--path=./install-here"],
            env={},
        )

        assert response == "string output"

    def test_json_format(self, mocker):
        mocker.patch("guardrails.cli.hub.install.os.environ", return_value={})
        mock_logger_debug = mocker.patch("guardrails.cli.hub.install.logger.debug")

        mock_sys_executable = mocker.patch("guardrails.cli.hub.install.sys.executable")

        mock_subprocess_check_output = mocker.patch(
            "guardrails.cli.hub.install.subprocess.check_output"
        )
        mock_subprocess_check_output.return_value = str.encode("json output")

        class MockBytesHeaderParser:
            def parsebytes(self, *args):
                return {"output": "json"}

        mock_bytes_parser = mocker.patch("guardrails.cli.hub.utils.BytesHeaderParser")
        mock_bytes_header_parser = MockBytesHeaderParser()
        mock_bytes_parser.return_value = mock_bytes_header_parser

        from guardrails.cli.hub.install import pip_process

        response = pip_process("show", "pip", format="json")

        assert mock_logger_debug.call_count == 3
        debug_calls = [
            call("running pip show  pip"),
            call("decoding output from pip show pip"),
            call(
                "JSON parse exception in decoding output from pip show pip. Falling back to accumulating the byte stream"  # noqa
            ),
        ]
        mock_logger_debug.assert_has_calls(debug_calls)

        mock_subprocess_check_output.assert_called_once_with(
            [mock_sys_executable, "-m", "pip", "show", "pip"], env={}
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
    mock_pip_process = mocker.patch("guardrails.cli.hub.utils.pip_process")
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


def test_quiet_install(mocker):
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
    install_hub_module(manifest, site_packages, quiet=True)

    mock_get_install_url.assert_called_once_with(manifest)
    mock_get_hub_directory.assert_called_once_with(manifest, site_packages)

    assert mock_pip_process.call_count == 5
    pip_calls = [
        call(
            "install",
            "mock-install-url",
            ["--target=mock/install/directory", "--no-deps", "-q"],
            quiet=True,
        ),
        call(
            "inspect",
            flags=["--path=mock/install/directory"],
            format="json",
            quiet=True,
            no_color=True,
        ),
        call("install", "rstr", quiet=True),
        call("install", "openai<2", quiet=True),
        call("install", "pydash>=7.0.6,<8.0.0", quiet=True),
    ]
    mock_pip_process.assert_has_calls(pip_calls)
