from unittest.mock import call
import pytest

from guardrails.cli.server.module_manifest import ModuleManifest, ModuleTags, Repository
from tests.unit_tests.mocks.mock_file import MockFile


def test_install_exits_early_if_uri_is_not_valid(mocker):
    mock_logger_error = mocker.patch("guardrails.cli.hub.install.logger.error")
    
    from guardrails.cli.hub.install import install, sys
    sys_exit_spy = mocker.spy(sys, "exit")

    with pytest.raises(SystemExit):
        install("not a hub uri")
    
    mock_logger_error.assert_called_once_with("Invalid URI!")
    sys_exit_spy.assert_called_once_with(1)


def test_install(mocker):
    mock_logger_log = mocker.patch("guardrails.cli.hub.install.logger.log")
    
    mock_fetch_module = mocker.patch("guardrails.cli.hub.install.fetch_module")
    manifest = ModuleManifest(
        "me",
        [],
        Repository(url="some-repo"),
        "guardrails",
        "test-validator",
        "test_validator",
        ["TestValidator"],
        ModuleTags()
    )
    mock_fetch_module.return_value = manifest

    mock_get_site_packages_location = mocker.patch("guardrails.cli.hub.install.get_site_packages_location")
    site_packages = "./.venv/lib/python3.X/site-packages"
    mock_get_site_packages_location.return_value = site_packages
    
    mock_install_hub_module = mocker.patch("guardrails.cli.hub.install.install_hub_module")
    mock_run_post_install = mocker.patch("guardrails.cli.hub.install.run_post_install")
    mock_add_to_hub_init = mocker.patch("guardrails.cli.hub.install.add_to_hub_init")

    from guardrails.cli.hub.install import install
    
    install("hub://guardrails/test-validator")

    log_calls = [
        call(level=25, msg=f"Installing hub://guardrails/test-validator..."),
        call(level=35, msg='\n    \n    Successfully installed guardrails/test-validator!\n\n    To use it in your python project, run:\n\n    from guardrails.hub import TestValidator\n    ')  # noqa
    ]
    assert mock_logger_log.call_count == 2
    mock_logger_log.assert_has_calls(log_calls)

    mock_fetch_module.assert_called_once_with("guardrails/test-validator")
    
    assert mock_get_site_packages_location.call_count == 1

    mock_install_hub_module.assert_called_once_with(manifest, site_packages)

    mock_run_post_install.assert_called_once_with(manifest)

    mock_add_to_hub_init.assert_called_once_with(manifest, site_packages)


def test_pip_process_no_package_string_format(mocker):
    mock_logger_debug = mocker.patch("guardrails.cli.hub.install.logger.debug")

    mock_sys_executable = mocker.patch("guardrails.cli.hub.install.sys.executable")
    
    mock_subprocess_check_output = mocker.patch("guardrails.cli.hub.install.subprocess.check_output")
    mock_subprocess_check_output.return_value = str.encode("string output")

    from guardrails.cli.hub.install import pip_process
    
    response = pip_process("inspect", flags=[f"--path=./install-here"])

    assert mock_logger_debug.call_count == 2
    debug_calls = [
        call("running pip inspect --path=./install-here "),
        call("decoding output from pip inspect ")
    ]
    mock_logger_debug.assert_has_calls(debug_calls)

    mock_subprocess_check_output.assert_called_once_with([mock_sys_executable, "-m", "pip", "inspect", "--path=./install-here"])

    assert response == "string output"


def test_pip_process_json_format(mocker):
    mock_logger_debug = mocker.patch("guardrails.cli.hub.install.logger.debug")

    mock_sys_executable = mocker.patch("guardrails.cli.hub.install.sys.executable")
    
    mock_subprocess_check_output = mocker.patch("guardrails.cli.hub.install.subprocess.check_output")
    mock_subprocess_check_output.return_value = str.encode("json output")

    class MockBytesHeaderParser:
        def parsebytes(self, *args):
            return { "output": "json" }
    mock_bytes_parser = mocker.patch("guardrails.cli.hub.install.BytesHeaderParser")
    mock_bytes_header_parser = MockBytesHeaderParser()
    mock_bytes_parser.return_value = mock_bytes_header_parser


    from guardrails.cli.hub.install import pip_process
    
    response = pip_process("show", "pip", format="json")

    assert mock_logger_debug.call_count == 2
    debug_calls = [
        call("running pip show  pip"),
        call("decoding output from pip show pip")
    ]
    mock_logger_debug.assert_has_calls(debug_calls)

    mock_subprocess_check_output.assert_called_once_with([mock_sys_executable, "-m", "pip", "show", "pip"])

    assert response == { "output": "json" }


def test_pip_process_called_process_error(mocker):
    mock_logger_error = mocker.patch("guardrails.cli.hub.install.logger.error")
    mock_logger_debug = mocker.patch("guardrails.cli.hub.install.logger.debug")
    mock_sys_executable = mocker.patch("guardrails.cli.hub.install.sys.executable")
    mock_subprocess_check_output = mocker.patch("guardrails.cli.hub.install.subprocess.check_output")
    
    from subprocess import CalledProcessError
    mock_subprocess_check_output.side_effect = CalledProcessError(1, "something")

    from guardrails.cli.hub.install import pip_process, sys
    sys_exit_spy = mocker.spy(sys, "exit")

    with pytest.raises(SystemExit):
        pip_process("inspect")

        mock_logger_debug.assert_called_once_with("running pip inspect  ")

        mock_subprocess_check_output.assert_called_once_with([mock_sys_executable, "-m", "pip", "inspect"])

        mock_logger_error.assert_called_once_with(f"Failed to inspect \nExit code: 1\nstdout: ")

        sys_exit_spy.assert_called_once_with(1)


def test_pip_process_other_exception(mocker):
    error = ValueError("something went wrong")
    mock_logger_debug = mocker.patch("guardrails.cli.hub.install.logger.debug")
    mock_logger_debug.side_effect = error

    mock_logger_error = mocker.patch("guardrails.cli.hub.install.logger.error")

    from guardrails.cli.hub.install import pip_process, sys
    sys_exit_spy = mocker.spy(sys, "exit")

    with pytest.raises(SystemExit):
        pip_process("inspect")

        mock_logger_debug.assert_called_once_with("running pip inspect  ")

        mock_logger_error.assert_called_once_with("An unexpected exception occurred while try to inspect !", error)

        sys_exit_spy.assert_called_once_with(1)


def test_get_site_packages_location(mocker):
    mock_pip_process = mocker.patch("guardrails.cli.hub.install.pip_process")
    mock_pip_process.return_value = { "Location": "/site-pacakges" }

    from guardrails.cli.hub.install import get_site_packages_location

    response = get_site_packages_location()

    mock_pip_process.assert_called_once_with("show", "pip", format="json")

    assert response == "/site-pacakges"


@pytest.mark.parametrize(
        "manifest,expected",
        [
            (
                ModuleManifest(
                    "me",
                    [],
                    Repository(url="some-repo"),
                    "guardrails-ai",
                    "test-validator",
                    "test_validator",
                    ["TestValidator"],
                    ModuleTags()
                ),
                ["guardrails_ai", "test_validator"]
            ),
            (
                ModuleManifest(
                    "me",
                    [],
                    Repository(url="some-repo"),
                    "",
                    "test-validator",
                    "test_validator",
                    ["TestValidator"],
                    ModuleTags()
                ),
                ["test_validator"]
            )
        ]
)
def test_get_org_and_package_dirs(manifest, expected):
    from guardrails.cli.hub.install import get_org_and_package_dirs

    actual = get_org_and_package_dirs(manifest)

    assert actual == expected


def test_get_hub_directory():
    manifest = ModuleManifest(
        "me",
        [],
        Repository(url="some-repo"),
        "guardrails-ai",
        "test-validator",
        "test_validator",
        ["TestValidator"],
        ModuleTags()
    )

    from guardrails.cli.hub.install import get_hub_directory
    hub_dir = get_hub_directory(manifest, "./site-packages")

    assert hub_dir == "./site-packages/guardrails/hub/guardrails_ai/test_validator"


def test_add_to_hub_init_closes_early_if_already_added(mocker):
    manifest = ModuleManifest(
        "me",
        [],
        Repository(url="some-repo"),
        "guardrails-ai",
        "test-validator",
        "validator",
        ["TestValidator", "helper"],
        ModuleTags()
    )
    site_packages = "./site-packages"

    hub_init_file = MockFile()
    ns_init_file = MockFile()
    mock_open = mocker.patch("guardrails.cli.hub.install.open")
    mock_open.side_effect = [hub_init_file, ns_init_file]

    mock_hub_read = mocker.patch.object(hub_init_file, "read")
    mock_hub_read.return_value = "from guardrails.hub.guardrails_ai.test_validator.validator import helper, TestValidator"

    hub_seek_spy = mocker.spy(hub_init_file, "seek")
    hub_write_spy = mocker.spy(hub_init_file, "write")
    hub_close_spy = mocker.spy(hub_init_file, "close")
    
    mock_ns_read = mocker.patch.object(ns_init_file, "read")
    mock_ns_read.return_value = "from guardrails.hub.guardrails_ai.test_validator.validator import helper, TestValidator"
    
    ns_seek_spy = mocker.spy(ns_init_file, "seek")
    ns_write_spy = mocker.spy(ns_init_file, "write")
    ns_close_spy = mocker.spy(ns_init_file, "close")

    mock_is_file = mocker.patch("guardrails.cli.hub.install.os.path.isfile")
    mock_is_file.return_value = True

    from guardrails.cli.hub.install import add_to_hub_init

    add_to_hub_init(manifest, site_packages)

    assert mock_open.call_count == 2
    open_calls = [
        call("./site-packages/guardrails/hub/__init__.py", "a+"),
        call("./site-packages/guardrails/hub/guardrails_ai/__init__.py", "a+")
    ]
    mock_open.assert_has_calls(open_calls)

    assert hub_seek_spy.call_count == 1
    assert mock_hub_read.call_count == 1
    assert hub_write_spy.call_count == 0
    assert hub_close_spy.call_count == 1


    mock_is_file.assert_called_once_with("./site-packages/guardrails/hub/guardrails_ai/__init__.py")
    assert ns_seek_spy.call_count == 1
    assert mock_ns_read.call_count == 1
    assert ns_write_spy.call_count == 0
    assert ns_close_spy.call_count == 1