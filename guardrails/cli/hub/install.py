import http.client
import json
import logging
import importlib
import os
import typer
import subprocess
import sys

from email.parser import BytesHeaderParser
from os.path import expanduser
from typing import List, Dict, Any, Literal, Optional, Union
from urllib.request import Request, urlopen
from guardrails.cli.hub.hub import hub
from guardrails.cli.logger import logger


hub_url = "https://raw.githubusercontent.com/guardrails-ai/guardrails-hub"
branch = "install-script"

string_format: Literal["string"] = "string"
json_format: Literal["json"] = "json"
minimum_guardrails_version: Literal["0.3.3"] = "0.3.3"

# TODO: Move this or find a library to do it
class SemVer:
    version: str
    major: int
    minor: int
    patch: int
    flag: Optional[str]

    def at(self, arr: List[str], index: int) -> Optional[str]:
        try:
            value = arr[index]
            return value
        except IndexError:
            return None
        
    def to_int(self, value: str) -> int:
        try:
            int_value = int(value)
            return int_value
        except TypeError:
            return 0

    def __init__(self, version: str):
        self.version = version
        versions = version.split(".")
        self.major = self.to_int(self.at(versions, 0))
        self.minor = self.to_int(self.at(versions, 1))
        patch = self.at(versions, 2)
        if patch:
            if patch.isnumeric():
                self.patch = patch 
            else:
                characters = [*patch]
                alpha = next(c for c in characters if c.isalpha())
                patch_n_flag = patch.split(alpha)
                self.patch = self.to_int(self.at(patch_n_flag, 0))
                self.flag = f"{alpha}{self.at(patch_n_flag, 1)}"

    def greater_or_equal(self, other_verison: str) -> bool:
        other_semver = SemVer(other_verison)
        if self.version == other_semver.version:
            return True
        elif self.major > other_semver.major:
            return True
        elif self.major < other_semver.major:
            return False
        # At this point we know the major versions are equal
        elif self.minor > other_semver.minor:
            return True
        elif self.minor < other_semver.minor:
            return False
        # At this point we know the minor versions are equal
        elif self.patch > other_semver.patch:
            return True
        elif self.patch < other_semver.patch:
            return False
        # At this point we know the patch versions are equal
        else:
            return True

# TODO: Move this or find a library to do it
class Credentials:
    client_id: str
    client_secret: str

    def __init__(self, client_id: str, client_secret: str):
        self.client_id = client_id
        self.client_secret = client_secret
    
    @classmethod
    def from_dict(cls, creds: Dict[str, str]):
        return cls(
            creds.get("client_id"),
            creds.get("client_secret")
        )

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


# TODO: Remove this; no longer necessary since this is in the guardrails package now
def verify_guardrails_installation(site_packages: str):
    guardrails_location = os.path.join(site_packages, "guardrails")
    guardrails_is_installed = os.path.exists(guardrails_location)
    if not guardrails_is_installed:
        logger.error("Guardrails must be installed to use the Guardrails Hub!")
        logger.info("Try running `python3 -m pip install guardrails-ai`")
        sys.exit(1)
    guardrails_details = pip_process("show", "guardrails-ai", format=json_format)
    guardrails_version = guardrails_details["Version"]
    
    if not SemVer(guardrails_version).greater_or_equal(minimum_guardrails_version):
        logger.error(f"You need guardrails-ai>={minimum_guardrails_version} to use the Guardrails Hub!")
        logger.info("Try running `python3 -m pip install guardrails-ai --upgrade`")
        sys.exit(1)


# NOTE: I don't like this but don't see another way without
        #  shimming the init file with all hub validators
def add_to_hub_init(manifest: dict):
    site_packages = get_site_packages_location()
    verify_guardrails_installation(site_packages)
    
    exports: List[str] = manifest.get("exports", [])
    sorted_exports = sorted(exports)
    module_name = manifest.get("module-name")
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


def run_post_install(manifest: dict):
    post_install_script = manifest.get("post-install")
    if post_install_script:
        module_name = manifest.get("module-name")
        post_install_module = post_install_script.removesuffix(".py")
        importlib.import_module(f"{module_name}.{post_install_module}")


def get_install_url(manifest: dict) -> str:
    repo = manifest.get("repository", {})
    repo_url = repo.get("url")
    branch = repo.get("branch")
    
    git_url = repo_url
    if not repo_url.startswith("git+"):
        git_url = f"git+{repo_url}"
    
    if branch is not None:
        git_url = f"{git_url}@{branch}"
    
    return git_url


# TODO: Error handling
def fetch_content(url: str):
    req = Request(url)
    # For Debugging
    # req.add_header("Cache-Control", "no-cache")
    conn = urlopen(req)
    contents = conn.read()
    conn.close()
    return json.loads(contents)


def fetch_module_manifest(module_summary: Dict[str, Any]) -> Dict[str, Any]:
    manifest_path = module_summary.get("manifest")
    manifest_url = f"{hub_url}/{branch}/{manifest_path}"
    return fetch_content(manifest_url)


def fetch_hub_index() -> List[Dict[str, Any]]:
    index_path = "index.json"
    index_url = f"{hub_url}/{branch}/{index_path}"
    return fetch_content(index_url)


def get_creds() -> Credentials:
    try:
        home = expanduser("~")
        guardrails_rc = os.path.join(home, '.guardrailsrc')
        with open(guardrails_rc) as rc_file:
            lines = rc_file.readlines()
            creds = {}
            for line in lines:
                key, value = line.split('=', 1)
                creds[key.strip()] = value.strip()
            return Credentials.from_dict(creds)

    except FileNotFoundError as e:
        logger.error(e)
        logger.error("Guardrails Hub credentials not found! Sign up to use the Hub here: {insert url}")
        sys.exit(1)

# This isn't secure
def auth_n():
    # This should stay here and the client credentials should be passed to our server as headers
    creds = get_creds()
    # The rest of this should happen on the server
    audience = "https://api.validator-hub.guardrailsai.com"
    conn = http.client.HTTPSConnection("guardrailsai.us.auth0.com")
    payload = json.dumps({ "client_id": creds.client_id, "client_secret": creds.client_secret, "audience": audience, "grant_type": "client_credentials" })
    headers = { 'content-type': "application/json" }
    conn.request("POST", "/oauth/token", payload, headers)

    res = conn.getresponse()
    data = json.loads(res.read().decode("utf-8"))
    if not data.get("access_token"):
        logger.error("Unauthorized!")
        sys.exit(1)
    


def get_module_name_arg() -> str:
    try:
        module_name = sys.argv[1]
        return module_name
    except IndexError:
        logger.error('You must specify the validator module you want to install!')
        logger.info('Example: `python3 install.py regex_match`')
        sys.exit(1)


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
    validator_mod_name = get_module_name_arg()

    ## ============ START: Move to server ============
    auth_n()

    # Discovery
    hub_index = fetch_hub_index()
    try:
        module_summary = next(x for x in hub_index if x.get("id") == validator_mod_name)
    except StopIteration:
        logger.error("Not Found!")
        logger.error(f"{validator_mod_name} does not exist in the hub!")
        sys.exit(1)

    module_manifest = fetch_module_manifest(module_summary)
    ## ============ END: Move to server ============

    install_url = get_install_url(module_manifest)

    # Install
    # TODO: Add target to install in org namespace
    download_output = pip_process("install", install_url)
    logger.info(download_output)

    # Post-install
    run_post_install(module_manifest)
    add_to_hub_init(module_manifest)