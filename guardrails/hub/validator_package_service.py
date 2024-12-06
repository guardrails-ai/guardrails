import importlib
import os
from pathlib import Path
import re
import subprocess
import sys

from typing import List, Literal, Optional
from types import ModuleType
from packaging.utils import canonicalize_name  # PEP 503

from guardrails.logger import logger as guardrails_logger


from guardrails.cli.hub.utils import PipProcessError, pip_process_with_custom_exception
from guardrails_hub_types import Manifest
from guardrails.cli.server.hub_client import get_validator_manifest
from guardrails.settings import settings


json_format: Literal["json"] = "json"
string_format: Literal["string"] = "string"


class ValidatorModuleType(ModuleType):
    __validator_exports__: List[str]


class FailedPackageInspection(Exception):
    pass


class FailedToLocateModule(Exception):
    pass


class FailedPackageInstallation(Exception):
    pass


class FailedPackageInstallationPostInstall(FailedPackageInstallation):
    pass


class InvalidHubInstallURL(Exception):
    pass


class ValidatorPackageService:
    @staticmethod
    def get_manifest_and_site_packages(module_name: str) -> tuple[Manifest, str]:
        module_manifest = get_validator_manifest(module_name)
        site_packages = ValidatorPackageService.get_site_packages_location()
        return (module_manifest, site_packages)

    @staticmethod
    def get_site_packages_location():
        pip_package_location = Path(ValidatorPackageService.get_module_path("pip"))
        # Get the location of site-packages
        site_packages_path = str(pip_package_location.parent)
        return site_packages_path

    @staticmethod
    def reload_module(module_path) -> ModuleType:
        try:
            reloaded_module = None
            # Dynamically import the module based on its path
            if "guardrails.hub" in sys.modules:
                # Reload the module if it's already imported
                importlib.reload(sys.modules["guardrails.hub"])
            if module_path not in sys.modules:
                # Import the module if it has not been imported yet
                reloaded_module = importlib.import_module(module_path)
                sys.modules[module_path] = reloaded_module
            else:
                reloaded_module = sys.modules[module_path]
            return reloaded_module
        except ModuleNotFoundError:
            raise
        except Exception:
            raise

    @staticmethod
    def get_validator_from_manifest(manifest: Manifest) -> ModuleType:
        """
        Get Validator class from the installed module based on the manifest.
        Note: manifest.exports yields a list of exported Validator classes.

        Args:
            manifest (Manifest): The manifest of the installed module

        Returns:
            Any: The Validator class from the installed module
        """

        validator_id = manifest.id
        import_path = ValidatorPackageService.get_import_path_from_validator_id(
            validator_id
        )

        import_line = f"{import_path}"

        # Reload or import the module
        return ValidatorPackageService.reload_module(import_line)

    @staticmethod
    def add_to_hub_inits(manifest: Manifest, site_packages: str):
        validator_id = manifest.id
        exports: List[str] = manifest.exports or []
        sorted_exports = sorted(exports, reverse=True)

        import_path = ValidatorPackageService.get_import_path_from_validator_id(
            validator_id
        )
        import_line = f"from {import_path} import {', '.join(sorted_exports)}"

        hub_init_location = os.path.join(
            site_packages, "guardrails", "hub", "__init__.py"
        )
        with open(hub_init_location, "a+") as hub_init:
            hub_init.seek(0, 0)
            content = hub_init.read()
            if import_line in content:
                hub_init.close()
            else:
                hub_init.seek(0, 2)
                if len(content) > 0:
                    hub_init.write("\n")
                hub_init.write(import_line)
                hub_init.close()

    @staticmethod
    def get_module_path(package_name):
        try:
            if package_name not in sys.modules:
                module = importlib.import_module(package_name)
                sys.modules[package_name] = module

            module = sys.modules[package_name]
            package_path = module.__path__[0]  # Take the first entry if it's a list

        except (ModuleNotFoundError, AttributeError, TypeError) as e:
            # wasn't able to import the module
            raise FailedToLocateModule(
                f"""
                    The module {package_name} could not be found in 
                    the current environment.
                """
            ) from e

        return package_path

    @staticmethod
    def get_validator_id(validator_uri: str):
        if not validator_uri.startswith("hub://"):
            raise InvalidHubInstallURL(
                "Invalid URI! The package URI must start with 'hub://'"
            )

        validator_uri_with_version = validator_uri.replace("hub://", "")

        validator_id_version_regex = (
            r"(?P<validator_id>[\/a-zA-Z0-9\-_]+)(?P<version>.*)"
        )
        match = re.match(validator_id_version_regex, validator_uri_with_version)
        validator_version = None

        if match:
            validator_id = match.group("validator_id")
            validator_version = (
                match.group("version").strip() if match.group("version") else None
            )
        else:
            validator_id = validator_uri_with_version

        return (validator_id, validator_version)

    @staticmethod
    def run_post_install(
        manifest: Manifest, site_packages: str, logger=guardrails_logger
    ):
        validator_id = manifest.id
        post_install_script = manifest.post_install

        if not post_install_script:
            return

        import_path = ValidatorPackageService.get_import_path_from_validator_id(
            validator_id
        )

        relative_path = os.path.join(
            site_packages,
            import_path,
            post_install_script,
        )

        if os.path.isfile(relative_path):
            try:
                logger.debug("running post install script...")
                command = [sys.executable, relative_path]
                subprocess.check_output(command)
            except subprocess.CalledProcessError as exc:
                logger.error(
                    (
                        f"Failed to run post install script for {manifest.id}\n"
                        f"Exit code: {exc.returncode}\n"
                        f"stdout: {exc.output}"
                    )
                )
                raise FailedPackageInstallationPostInstall(
                    f"Failed to run post install script for {manifest.id}\n"
                )
            except Exception as e:
                logger.error(
                    f"An unexpected exception occurred while running the post install script for {manifest.id}!",  # noqa
                    e,
                )
                raise FailedPackageInstallationPostInstall(
                    f"""
                    An unexpected exception occurred while running the post install 
                    script for {manifest.id}!
                    """
                )

    @staticmethod
    def get_normalized_package_name(validator_id: str):
        validator_id_parts = validator_id.split("/")
        concatanated_package_name = (
            f"{validator_id_parts[0]}-grhub-{validator_id_parts[1]}"
        )
        pep_503_package_name = canonicalize_name(concatanated_package_name)
        return pep_503_package_name

    @staticmethod
    def get_import_path_from_validator_id(validator_id):
        pep_503_package_name = ValidatorPackageService.get_normalized_package_name(
            validator_id
        )
        return pep_503_package_name.replace("-", "_")

    @staticmethod
    def install_hub_module(
        validator_id: str,
        validator_version: Optional[str] = "",
        quiet: bool = False,
        upgrade: bool = False,
        logger=guardrails_logger,
    ):
        pep_503_package_name = ValidatorPackageService.get_normalized_package_name(
            validator_id
        )
        validator_version = validator_version if validator_version else ""

        guardrails_token = settings.rc.token

        pip_flags = [
            f"--index-url=https://__token__:{guardrails_token}@pypi.guardrailsai.com/simple",
            "--extra-index-url=https://pypi.org/simple",
        ]

        if upgrade:
            pip_flags.append("--upgrade")

        if quiet:
            pip_flags.append("-q")

        # Install from guardrails hub pypi server with public pypi index as fallback

        try:
            full_package_name = f"{pep_503_package_name}[validators]{validator_version}"
            download_output = pip_process_with_custom_exception(
                "install", full_package_name, pip_flags, quiet=quiet
            )
            if not quiet:
                logger.info(download_output)
        except PipProcessError:
            try:
                full_package_name = f"{pep_503_package_name}{validator_version}"
                download_output = pip_process_with_custom_exception(
                    "install", full_package_name, pip_flags, quiet=quiet
                )
                if not quiet:
                    logger.info(download_output)
            except PipProcessError as e:
                action = e.action
                package = e.package
                stderr = e.stderr
                stdout = e.stdout
                returncode = e.returncode
                logger.error(
                    (
                        f"Failed to {action} {package}\n"
                        f"Exit code: {returncode}\n"
                        f"stderr: {(stderr or '').strip()}\n"
                        f"stdout: {(stdout or '').strip()}"
                    )
                )
                raise
            except Exception as e:
                logger.error(
                    "An unexpected exception occurred while "
                    f"installing {validator_id}: ",
                    e,
                )
                raise
