import json
import os
import subprocess
import sys
from email.parser import BytesHeaderParser
from typing import List, Literal, Union
from pydash.strings import snake_case

from guardrails.cli.server.module_manifest import ModuleManifest
from guardrails.cli.logger import logger


json_format: Literal["json"] = "json"
string_format: Literal["string"] = "string"


def pip_process(
    action: str,
    package: str = "",
    flags: List[str] = [],
    format: Union[Literal["string"], Literal["json"]] = string_format,
    quiet: bool = False
) -> Union[str, dict]:
    try:
        command = [sys.executable, "-m", "pip", action] + flags
        if package:
            command.append(package)
        
        stderr = subprocess.DEVNULL if quiet else None
        
        output = subprocess.check_output(command, stderr=stderr)
        
        if format == json_format:
            parsed = BytesHeaderParser().parsebytes(output)
            try:
                return json.loads(str(parsed))
            except Exception:
                if not quiet:
                    logger.debug(
                        f"JSON parse exception in decoding output from pip {action} {package}. Falling back to accumulating the byte stream",
                    )
                accumulator = {}
                for key, value in parsed.items():
                    accumulator[key] = value
                return accumulator
        return output.decode()
    except subprocess.CalledProcessError as exc:
        if not quiet:
            logger.error(
                f"Failed to {action} {package}\nExit code: {exc.returncode}\nstdout: {exc.output.decode()}"
            )
        sys.exit(1)
    except Exception as e:
        if not quiet:
            logger.error(
                f"An unexpected exception occurred while trying to {action} {package}: {str(e)}"
            )
        sys.exit(1)


def get_site_packages_location() -> str:
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
