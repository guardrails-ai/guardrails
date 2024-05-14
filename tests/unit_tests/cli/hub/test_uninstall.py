import pytest
from guardrails.cli.server.module_manifest import ModuleManifest


# Define a function to create a basic ModuleManifest instance for testing
def create_test_manifest():
    return ModuleManifest(
        id="id",
        name="name",
        author={"name": "me", "email": "me@me.me"},
        maintainers=[],
        repository={"url": "some-repo"},
        namespace="guardrails",
        package_name="test-validator",
        module_name="test_validator",
        exports=["TestValidator"],
        tags={},
    )


# Define tests for uninstalling a hub module
class TestUninstall:
    @pytest.fixture
    def setup_manifest_and_site_packages(self, mocker):
        manifest = create_test_manifest()
        mocker.patch("uninstall.get_validator_manifest", return_value=manifest)
        mocker.patch(
            "uninstall.get_site_packages_location", return_value="/fake/site-packages"
        )
        return manifest

    def test_uninstall_invalid_uri(self, mocker):
        # Setup mocks
        mock_exit = mocker.patch("uninstall.sys.exit")
        mock_logger = mocker.patch("uninstall.logger.error")

        from guardrails.cli.hub.uninstall import uninstall

        uninstall("not a hub uri")

        mock_logger.assert_called_once_with("Invalid URI!")
        mock_exit.assert_called_once_with(1)

    def test_uninstall_successful(self, setup_manifest_and_site_packages, mocker):
        # Setup mocks
        mock_uninstall_hub_module = mocker.patch("uninstall.uninstall_hub_module")
        mock_remove_from_hub_inits = mocker.patch("uninstall.remove_from_hub_inits")
        mock_console = mocker.patch("uninstall.console.print")
        mock_logger = mocker.patch("uninstall.logger.log")

        from guardrails.cli.hub.uninstall import uninstall

        uninstall("hub://guardrails/test-validator")

        mock_uninstall_hub_module.assert_called_once_with(
            setup_manifest_and_site_packages, "/fake/site-packages"
        )
        mock_remove_from_hub_inits.assert_called_once_with(
            setup_manifest_and_site_packages, "/fake/site-packages"
        )

        assert mock_console.call_count == 2
        mock_console.assert_called_with("✅ Successfully uninstalled!")
        mock_logger.assert_called()

    def test_uninstall_failures(self, setup_manifest_and_site_packages, mocker):
        # Simulate an error during uninstall
        mocker.patch(
            "uninstall.uninstall_hub_module",
            side_effect=Exception("Test error during uninstall"),
        )
        mock_exit = mocker.patch("uninstall.sys.exit")
        mock_console = mocker.patch("uninstall.console.print")
        mock_logger = mocker.patch("uninstall.logger.error")

        from guardrails.cli.hub.uninstall import uninstall

        with pytest.raises(Exception):
            uninstall("hub://guardrails/test-validator")

        mock_logger.assert_called_with(
            "Failed during uninstall: Test error during uninstall"
        )
        mock_console.assert_not_called()
        mock_exit.assert_not_called()
