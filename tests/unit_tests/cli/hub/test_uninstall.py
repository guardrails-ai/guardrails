from unittest.mock import mock_open, call

import pytest

from guardrails.cli.server.module_manifest import ModuleManifest
from guardrails.cli.hub.uninstall import remove_from_hub_inits

manifest_mock = ModuleManifest(
    encoder="some_encoder",
    id="module_id",
    name="test_module",
    author={"name": "Author Name", "email": "author@example.com"},
    maintainers=[{"name": "Maintainer Name", "email": "maintainer@example.com"}],
    repository={"url": "https://github.com/example/repo"},
    namespace="guardrails",
    package_name="test_package",
    module_name="test_module",
    exports=["Validator", "Helper"],
)


def test_remove_from_hub_inits(mocker):
    mocker.patch(
        "guardrails.cli.hub.uninstall.get_org_and_package_dirs",
        return_value=["guardrails", "test_package"],
    )
    mock_remove_line = mocker.patch("guardrails.cli.hub.uninstall.remove_line")
    mock_remove_dirs = mocker.patch("os.removedirs")

    remove_from_hub_inits(manifest_mock, "/site-packages")

    expected_calls = [
        call(
            "/site-packages/guardrails/hub/__init__.py",
            "from guardrails.hub.guardrails.test_package.test_module import "
            "Validator, Helper",
        ),
        call(
            "/site-packages/guardrails/hub/guardrails/__init__.py",
            "from guardrails.hub.guardrails.test_package.test_module import "
            "Validator, Helper",
        ),
    ]

    mock_remove_line.assert_has_calls(expected_calls, any_order=True)
    mock_remove_dirs.assert_called_once_with("/site-packages/guardrails/hub/guardrails")


def test_uninstall_invalid_uri(mocker):
    with pytest.raises(SystemExit):
        mock_logger_error = mocker.patch("guardrails.cli.hub.uninstall.logger.error")
        mocker.patch(
            "guardrails.cli.hub.uninstall.get_validator_manifest",
            return_value=manifest_mock,
        )
        mocker.patch(
            "guardrails.cli.hub.utils.pip_process",
            return_value={"Location": "/fake/site-packages"},
        )

        m_open = mock_open(read_data="import something")
        mocker.patch("builtins.open", m_open)
        mocker.patch("os.path.exists", return_value=True)

        mocker.patch("subprocess.check_call")
        from guardrails.cli.hub.uninstall import uninstall

        uninstall("not a hub uri")

        mock_logger_error.assert_called_once_with("Invalid URI!")

        m_open.assert_called()

        mock_subprocess_check_call = mocker.patch("subprocess.check_call")
        mock_subprocess_check_call.assert_not_called()


def test_uninstall_valid_uri(mocker):
    mocker.patch(
        "guardrails.cli.hub.uninstall.get_validator_manifest",
        return_value=manifest_mock,
    )
    mocker.patch(
        "guardrails.cli.hub.uninstall.get_site_packages_location",
        return_value="/site-packages",
    )
    mock_uninstall_hub_module = mocker.patch(
        "guardrails.cli.hub.uninstall.uninstall_hub_module"
    )
    mock_remove_from_hub_inits = mocker.patch(
        "guardrails.cli.hub.uninstall.remove_from_hub_inits"
    )
    mocker.patch("guardrails.cli.hub.uninstall.console")

    from guardrails.cli.hub.uninstall import uninstall

    uninstall("hub://guardrails/test-validator")

    mock_uninstall_hub_module.assert_called_once_with(manifest_mock, "/site-packages")
    mock_remove_from_hub_inits.assert_called_once_with(manifest_mock, "/site-packages")
