import importlib
import os
import typer
import subprocess
import sys

from email.parser import BytesHeaderParser
from typing import List, Literal, Optional, Union
from guardrails.cli.hub.hub import hub
from guardrails.cli.logger import logger
from guardrails.cli.server.hub_client import fetch_module
from guardrails.cli.server.module_manifest import ModuleManifest


string_format: Literal["string"] = "string"
json_format: Literal["json"] = "json"


def pip_process(
    action: str,
    package: str,
    flags: Optional[List[str]] = [],
    format: Optional[Union[string_format, json_format]] = string_format
):
    try:
        logger.debug(f"running pip {action} {' '.join(flags)} {package}")
        command = [sys.executable, "-m", "pip", action]
        command.extend(flags)
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


# NOTE: I don't like this but don't see another way without
        #  shimming the init file with all hub validators
def add_to_hub_init(manifest: ModuleManifest):
    site_packages = get_site_packages_location()
    
    exports: List[str] = manifest.exports or []
    sorted_exports = sorted(exports)
    module_name = manifest.module_name
    import_line = f"from {module_name} import {', '.join(sorted_exports)}"

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
        return


def run_post_install(manifest: ModuleManifest):
    post_install_script = manifest.post_install
    if post_install_script:
        module_name = manifest.module_name
        post_install_module = post_install_script.removesuffix(".py")
        importlib.import_module(f"{module_name}.{post_install_module}")


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
    logger.info(f"Installing {package_uri}...")
    # Validation
    module_name = package_uri.replace("hub://", "")

    module_manifest = fetch_module(module_name)

    install_url = get_install_url(module_manifest)

    # Install
    # TODO: Add target to install in org namespace
    download_output = pip_process("install", install_url)
    logger.info(download_output)

    # Post-install
    run_post_install(module_manifest)
    add_to_hub_init(module_manifest)