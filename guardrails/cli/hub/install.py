import importlib
import json
import os
import typer
import subprocess
import sys

from email.parser import BytesHeaderParser
from pydash.strings import snake_case
from typing import List, Literal, Optional, Union
from string import Template
from guardrails.cli.hub.hub import hub
from guardrails.cli.logger import logger, LEVELS
from guardrails.cli.server.hub_client import fetch_module
from guardrails.cli.server.module_manifest import ModuleManifest
from guardrails.classes.generic import Stack


string_format: Literal["string"] = "string"
json_format: Literal["json"] = "json"


def pip_process(
    action: str,
    package: Optional[str] = '',
    flags: Optional[List[str]] = [],
    format: Optional[Union[string_format, json_format]] = string_format
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
    pip_location = output["Location"]
    return pip_location


def get_org_and_package_dirs(manifest: ModuleManifest) -> List[str]:
    org_name = manifest.namespace
    package_name = manifest.package_name
    org = snake_case(org_name if len(org_name) > 1 else '')
    package = snake_case(package_name if len(package_name) > 1 else package_name)
    return list(filter(None, [org, package]))


def get_hub_directory(manifest: ModuleManifest, site_packages: str) -> str:
    org_package = get_org_and_package_dirs(manifest)
    return os.path.join(site_packages, 'guardrails', 'hub', *org_package)


# NOTE: I don't like this but don't see another way without
        #  shimming the init file with all hub validators
def add_to_hub_init(manifest: ModuleManifest, site_packages: str):
    org_package = get_org_and_package_dirs(manifest)
    exports: List[str] = manifest.exports or []
    sorted_exports = sorted(exports)
    module_name = manifest.module_name
    relative_path = '.'.join([*org_package, module_name])
    import_line = f"from guardrails.hub.{relative_path} import {', '.join(sorted_exports)}"

    hub_init_location = os.path.join(site_packages, 'guardrails', 'hub', '__init__.py')
    with open(hub_init_location, 'a+') as hub_init:
        hub_init.seek(0, 0)
        content = hub_init.read()
        if import_line in content:
            hub_init.close()
            return
        hub_init.seek(0, 2)
        if len(content) > 0:
            hub_init.write("\n")
        hub_init.write(import_line)
        hub_init.close()

    namespace = org_package[0]
    namespace_init_location = os.path.join(site_packages, 'guardrails', 'hub', namespace, '__init__.py')
    if os.path.isfile(namespace_init_location):
        with open(namespace_init_location, 'a+') as namespace_init:
            namespace_init.seek(0, 0)
            content = hub_init.read()
            if import_line in content:
                namespace_init.close()
                return
            namespace_init.seek(0, 2)
            if len(content) > 0:
                namespace_init.write("\n")
            namespace_init.write(import_line)
            namespace_init.close()
            return
    else:
        with open(namespace_init_location, 'w') as namespace_init:
            namespace_init.write(import_line)
            namespace_init.close()
            return


def run_post_install(manifest: ModuleManifest):
    org_package = get_org_and_package_dirs(manifest)
    post_install_script = manifest.post_install
    if post_install_script:
        module_name = manifest.module_name
        post_install_module = post_install_script.removesuffix(".py")
        relative_path = '.'.join([*org_package, module_name])
        importlib.import_module(f"guardrails.hub.{relative_path}.{post_install_module}")


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
    download_output = pip_process("install", install_url, [f"--target={install_directory}", "--no-deps"])
    logger.info(download_output)

    # Install validator module's dependencies in normal site-packages directory
    inspect_output = pip_process("inspect", flags=[f"--path={install_directory}"], format=json_format)
    inspection: dict = json.loads(str(inspect_output))
    dependencies = Stack(*inspection.get("installed", [])).at(0, {}).get("metadata", {}).get("requires_dist", [])
    requirements = filter(lambda dep: "extra" not in dep, dependencies)
    for req in requirements:
        req_info = Stack(*req.split(" "))
        name = req_info.at(0, "").strip()
        versions = req_info.at(1, "").strip("()")
        if name:
            install_spec = (
                name
                if not versions
                else f"{name}{versions}"
            )
            dep_install_output = pip_process("install", install_spec)
            logger.info(dep_install_output)


@hub.command()
def install(
    package_uri: str = typer.Argument(
        help="URI to the package to install. Example: hub://guardrails/regex-match."
    )
):
    """Install a validator from the Hub."""
    if not package_uri.startswith("hub://"):
        logger.error("Invalid URI!")
        sys.exit(1)
    logger.log(level=LEVELS.get("NOTICE"), msg=f"Installing {package_uri}...")
    # Validation
    module_name = package_uri.replace("hub://", "")

    # Prep
    module_manifest = fetch_module(module_name)
    site_packages = get_site_packages_location()

    # Install
    install_hub_module(module_manifest, site_packages)
    
    # Post-install
    run_post_install(module_manifest)
    add_to_hub_init(module_manifest, site_packages)
    success_message = Template("""
    
    Successfully installed ${module_name}!

    To use it in your python project, run:

    from guardrails.hub import ${export_name}
    """).safe_substitute({ "module_name": module_name, "export_name": module_manifest.exports[0] })
    logger.log(level=LEVELS.get("SUCCESS"), msg=success_message)