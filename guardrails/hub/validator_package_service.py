from contextlib import contextmanager
import importlib
import os
from pathlib import Path
import subprocess
import sys
from string import Template

from typing import List, Literal, Callable
from pydash.strings import snake_case

from guardrails.classes.credentials import Credentials
from guardrails.classes.generic.stack import Stack
from guardrails.logger import logger as guardrails_logger
from guardrails.cli.logger import LEVELS, logger as cli_logger


from guardrails.cli.hub.utils import pip_process
from guardrails.cli.server.module_manifest import ModuleManifest
from guardrails.cli.server.hub_client import get_validator_manifest
from guardrails.cli.hub.console import console


json_format: Literal["json"] = "json"
string_format: Literal["string"] = "string"


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


@contextmanager
def do_nothing_context(*args, **kwargs):
    try:
        yield
    finally:
        pass


class ValidatorPackageService:
    @staticmethod
    def install(
        package_uri: str,
        install_local_models=None,
        quiet: bool = True,
        install_local_models_confirm: Callable = lambda: True,
    ):
        """Install a validator from the Hub."""

        verbose_printer = console.print
        quiet_printer = console.print if not quiet else lambda x: None

        # 1. Validation
        has_rc_file = Credentials.has_rc_file()
        module_name = ValidatorPackageService.get_module_name(package_uri)

        installing_msg = f"Installing {package_uri}..."
        cli_logger.log(
            level=LEVELS.get("SPAM"),  # type: ignore
            msg=installing_msg,
        )
        verbose_printer(installing_msg)

        # Define Loader for UX purposes
        loader = console.status if not quiet else do_nothing_context

        # 2. Prep Installation
        fetch_manifest_msg = "Fetching manifest"
        with loader(fetch_manifest_msg, spinner="bouncingBar"):
            (module_manifest, site_packages) = ValidatorPackageService.install__prep(
                module_name
            )

        # 3. Install - Pip Installation of git module
        dl_deps_msg = "Downloading dependencies"
        with loader(dl_deps_msg, spinner="bouncingBar"):
            ValidatorPackageService.install_hub_module(
                module_manifest, site_packages, quiet=quiet, logger=cli_logger
            )

        use_remote_endpoint = False
        module_has_endpoint = (
            module_manifest.tags and module_manifest.tags.has_guardrails_endpoint
        )

        try:
            if has_rc_file:
                # if we do want to remote then we don't want to install local models
                use_remote_endpoint = (
                    Credentials.from_rc_file(cli_logger).use_remote_inferencing
                    and module_has_endpoint
                )
            elif install_local_models is None and module_has_endpoint:
                install_local_models = install_local_models_confirm()
        except AttributeError:
            pass

        # 4. Post Installation
        install_local_models = (
            install_local_models if install_local_models is not None else True
        )
        if not use_remote_endpoint and install_local_models is True:
            cli_logger.log(
                level=LEVELS.get("SPAM"),  # type: ignore
                msg="Installing models locally!",
            )
            post_msg = "Running post-install setup"
            with loader(post_msg, spinner="bouncingBar"):
                ValidatorPackageService.run_post_install(
                    module_manifest, site_packages, logger=cli_logger
                )
        else:
            cli_logger.log(
                level=LEVELS.get("SPAM"),  # type: ignore
                msg="Skipping post install, models will not be "
                "downloaded for local inference.",
            )
        ValidatorPackageService.add_to_hub_inits(module_manifest, site_packages)

        # 5. Get Validator Class for the installed module
        validators = ValidatorPackageService.get_validator_from_manifest(
            module_manifest
        )

        # Print success messages
        cli_logger.info("Installation complete")

        verbose_printer(f"✅Successfully installed {module_name}!\n\n")
        success_message_cli = Template(
            "[bold]Import validator:[/bold]\n"
            "from guardrails.hub import ${export}\n\n"
            "[bold]Get more info:[/bold]\n"
            "https://hub.guardrailsai.com/validator/${id}\n"
        ).safe_substitute(
            module_name=package_uri,
            id=module_manifest.id,
            export=module_manifest.exports[0],
        )
        success_message_logger = Template(
            "✅Successfully installed ${module_name}!\n\n"
            "Import validator:\n"
            "from guardrails.hub import ${export}\n\n"
            "Get more info:\n"
            "https://hub.guardrailsai.com/validator/${id}\n"
        ).safe_substitute(
            module_name=package_uri,
            id=module_manifest.id,
            export=module_manifest.exports[0],
        )
        quiet_printer(success_message_cli)  # type: ignore
        cli_logger.log(level=LEVELS.get("SPAM"), msg=success_message_logger)  # type: ignore

        return validators

    @staticmethod
    def install__prep(module_name: str) -> tuple[ModuleManifest, str]:
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
    def reload_module(module_path):
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
    def get_validator_from_manifest(manifest: ModuleManifest):
        org_package = ValidatorPackageService.get_org_and_package_dirs(manifest)
        module_name = manifest.module_name

        _relative_path = ".".join([*org_package, module_name])

        exports: List[str] = manifest.exports or []
        import_line = f"guardrails.hub.{_relative_path}"
        print(f" => Import {import_line}")
        print(f" => Exports: {exports}")

        # Reload or import the module
        return ValidatorPackageService.reload_module(import_line)

    @staticmethod
    def get_org_and_package_dirs(
        manifest: ModuleManifest,
    ) -> List[str]:
        org_name = manifest.namespace
        package_name = manifest.package_name
        org = snake_case(org_name if len(org_name) > 1 else "")
        package = snake_case(package_name if len(package_name) > 1 else package_name)
        return list(filter(None, [org, package]))

    @staticmethod
    def add_to_hub_inits(manifest: ModuleManifest, site_packages: str):
        org_package = ValidatorPackageService.get_org_and_package_dirs(manifest)
        exports: List[str] = manifest.exports or []
        sorted_exports = sorted(exports, reverse=True)
        module_name = manifest.module_name
        relative_path = ".".join([*org_package, module_name])
        import_line = (
            f"from guardrails.hub.{relative_path} import {', '.join(sorted_exports)}"
        )

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

        namespace = org_package[0]
        namespace_init_location = os.path.join(
            site_packages, "guardrails", "hub", namespace, "__init__.py"
        )
        if os.path.isfile(namespace_init_location):
            with open(namespace_init_location, "a+") as namespace_init:
                namespace_init.seek(0, 0)
                content = namespace_init.read()
                if import_line in content:
                    namespace_init.close()
                else:
                    namespace_init.seek(0, 2)
                    if len(content) > 0:
                        namespace_init.write("\n")
                    namespace_init.write(import_line)
                    namespace_init.close()
        else:
            with open(namespace_init_location, "w") as namespace_init:
                namespace_init.write(import_line)
                namespace_init.close()

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
    def get_module_name(package_uri: str):
        if not package_uri.startswith("hub://"):
            raise InvalidHubInstallURL(
                "Invalid URI! The package URI must start with 'hub://'"
            )

        module_name = package_uri.replace("hub://", "")
        return module_name

    @staticmethod
    def get_install_url(manifest: ModuleManifest) -> str:
        repo = manifest.repository
        repo_url = repo.url
        branch = repo.branch

        git_url = repo_url
        if not repo_url.startswith("git+"):
            git_url = f"git+{repo_url}"

        if branch is not None:
            git_url = f"{git_url}@{branch}"

        return git_url

    @staticmethod
    def run_post_install(
        manifest: ModuleManifest, site_packages: str, logger=guardrails_logger
    ):
        org_package = ValidatorPackageService.get_org_and_package_dirs(manifest)
        post_install_script = manifest.post_install
        if not post_install_script:
            return

        module_name = manifest.module_name
        relative_path = os.path.join(
            site_packages,
            "guardrails",
            "hub",
            *org_package,
            module_name,
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
    def get_hub_directory(manifest: ModuleManifest, site_packages: str) -> str:
        org_package = ValidatorPackageService.get_org_and_package_dirs(manifest)
        return os.path.join(site_packages, "guardrails", "hub", *org_package)

    @staticmethod
    def install_hub_module(
        module_manifest: ModuleManifest,
        site_packages: str,
        quiet: bool = False,
        logger=guardrails_logger,
    ):
        install_url = ValidatorPackageService.get_install_url(module_manifest)
        install_directory = ValidatorPackageService.get_hub_directory(
            module_manifest, site_packages
        )

        pip_flags = [f"--target={install_directory}", "--no-deps"]
        if quiet:
            pip_flags.append("-q")

        # Install validator module in namespaced directory under guardrails.hub
        download_output = pip_process("install", install_url, pip_flags, quiet=quiet)
        if not quiet:
            logger.info(download_output)

        # Install validator module's dependencies in normal site-packages directory
        inspect_output = pip_process(
            "inspect",
            flags=[f"--path={install_directory}"],
            format=json_format,
            quiet=quiet,
            no_color=True,
        )

        # throw if inspect_output is a string. Mostly for pyright
        if isinstance(inspect_output, str):
            logger.error("Failed to inspect the installed package!")
            raise FailedPackageInspection

        dependencies = (
            Stack(*inspect_output.get("installed", []))
            .at(0, {})
            .get("metadata", {})  # type: ignore
            .get("requires_dist", [])  # type: ignore
        )
        requirements = list(filter(lambda dep: "extra" not in dep, dependencies))
        for req in requirements:
            if "git+" in req:
                install_spec = req.replace(" ", "")
                dep_install_output = pip_process("install", install_spec, quiet=quiet)
                if not quiet:
                    logger.info(dep_install_output)
            else:
                req_info = Stack(*req.split(" "))
                name = req_info.at(0, "").strip()  # type: ignore
                versions = req_info.at(1, "").strip("()")  # type: ignore
                if name:
                    install_spec = name if not versions else f"{name}{versions}"
                    dep_install_output = pip_process(
                        "install", install_spec, quiet=quiet
                    )
                    if not quiet:
                        logger.info(dep_install_output)
