import json
import os
import subprocess
import sys
from email.parser import BytesHeaderParser
from string import Template
from typing import List, Literal, Union

import typer
from pydash.strings import snake_case

from guardrails.classes.generic import Stack
from guardrails.cli.hub.hub import hub_command
from guardrails.cli.logger import LEVELS, logger
from guardrails.cli.server.hub_client import get_validator_manifest
from guardrails.cli.server.module_manifest import ModuleManifest


def removesuffix(string: str, suffix: str) -> str:
    if sys.version_info.minor >= 9:
        return string.removesuffix(suffix)  # type: ignore
    else:
        if string.endswith(suffix):
            return string[: -len(suffix)]
        return string


string_format: Literal["string"] = "string"
json_format: Literal["json"] = "json"


def pip_process(
    action: str,
    package: str = "",
    flags: List[str] = [],
    format: Union[Literal["string"], Literal["json"]] = string_format,
):
    try:
        logger.debug(f"running pip {action} {' '.join(flags)} {package}")
        command = [sys.executable, "-m", "pip", action]
        command.extend(flags)
        if package:
            command.append(package)
        output = subprocess.check_output(command)
        logger.debug(f"decoding output from pip {action} {package}")
        if format == json_format:
            return BytesHeaderParser().parsebytes(output)
        return str(output.decode())
    except subprocess.CalledProcessError as exc:
        logger.error(
            (
                f"Failed to {action} {package}\n"
                f"Exit code: {exc.returncode}\n"
                f"stdout: {exc.output}"
            )
        )
        sys.exit(1)
    except Exception as e:
        logger.error(
            f"An unexpected exception occurred while try to {action} {package}!",
            e,
        )
        sys.exit(1)


def get_site_packages_location():
    output = pip_process("show", "pip", format=json_format)
    pip_location = output["Location"]  # type: ignore
    return pip_location


def get_org_and_package_dirs(manifest: ModuleManifest) -> List[str]:
    org_name = manifest.namespace
    package_name = manifest.package_name
    org = snake_case(org_name if len(org_name) > 1 else "")
    package = snake_case(package_name if len(package_name) > 1 else package_name)
    return list(filter(None, [org, package]))


def get_hub_directory(manifest: ModuleManifest, site_packages: str) -> str:
    org_package = get_org_and_package_dirs(manifest)
    return os.path.join(site_packages, "guardrails", "hub", *org_package)


# NOTE: I don't like this but don't see another way without
#  shimming the init file with all hub validators
def add_to_hub_inits(manifest: ModuleManifest, site_packages: str):
    org_package = get_org_and_package_dirs(manifest)
    exports: List[str] = manifest.exports or []
    sorted_exports = sorted(exports, reverse=True)
    module_name = manifest.module_name
    relative_path = ".".join([*org_package, module_name])
    import_line = (
        f"from guardrails.hub.{relative_path} import {', '.join(sorted_exports)}"
    )

    hub_init_location = os.path.join(site_packages, "guardrails", "hub", "__init__.py")
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


def run_post_install(manifest: ModuleManifest, site_packages: str):
    org_package = get_org_and_package_dirs(manifest)
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
            sys.exit(1)
        except Exception as e:
            logger.error(
                f"An unexpected exception occurred while running the post install script for {manifest.id}!",  # noqa
                e,
            )
            sys.exit(1)


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


def install_hub_module(module_manifest: ModuleManifest, site_packages: str):
    install_url = get_install_url(module_manifest)
    install_directory = get_hub_directory(module_manifest, site_packages)

    # Install validator module in namespaced directory under guardrails.hub
    download_output = pip_process(
        "install", install_url, [f"--target={install_directory}", "--no-deps"]
    )
    logger.info(download_output)

    # Install validator module's dependencies in normal site-packages directory
    inspect_output = pip_process(
        "inspect", flags=[f"--path={install_directory}"], format=json_format
    )
    inspection: dict = json.loads(str(inspect_output))
    dependencies = (
        Stack(*inspection.get("installed", []))
        .at(0, {})
        .get("metadata", {})  # type: ignore
        .get("requires_dist", [])  # type: ignore
    )
    requirements = filter(lambda dep: "extra" not in dep, dependencies)
    for req in requirements:
        req_info = Stack(*req.split(" "))
        name = req_info.at(0, "").strip()  # type: ignore
        versions = req_info.at(1, "").strip("()")  # type: ignore
        if name:
            install_spec = name if not versions else f"{name}{versions}"
            dep_install_output = pip_process("install", install_spec)
            logger.info(dep_install_output)


@hub_command.command()
def install(
    package_uri: str = typer.Argument(
        help="URI to the package to install. Example: hub://guardrails/regex_match."
    ),
):
    """Install a validator from the Hub."""
    if not package_uri.startswith("hub://"):
        logger.error("Invalid URI!")
        sys.exit(1)
    logger.log(
        level=LEVELS.get("NOTICE"), msg=f"Installing {package_uri}..."  # type: ignore
    )
    # Validation
    module_name = package_uri.replace("hub://", "")

    # Prep
    module_manifest = get_validator_manifest(module_name)
    site_packages = get_site_packages_location()

    # Install
    install_hub_module(module_manifest, site_packages)

    # Post-install
    run_post_install(module_manifest, site_packages)
    add_to_hub_inits(module_manifest, site_packages)
    success_message = Template(
        """

    Successfully installed ${module_name}!

    See how to use it here: https://hub.guardrailsai.com/validator/${id}
    """
    ).safe_substitute(module_name=module_name, id=module_manifest.id)
    logger.log(level=LEVELS.get("SUCCESS"), msg=success_message)  # type: ignore
