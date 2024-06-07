import json
import os
import re
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
    quiet: bool = False,
    no_color: bool = False,
) -> Union[str, dict]:
    try:
        if not quiet:
            logger.debug(f"running pip {action} {' '.join(flags)} {package}")
        command = [sys.executable, "-m", "pip", action]
        command.extend(flags)
        if package:
            command.append(package)

        env = dict(os.environ)
        if no_color:
            env["NO_COLOR"] = "true"
        if not quiet:
            logger.debug(f"decoding output from pip {action} {package}")
            output = subprocess.check_output(command, env=env)
        else:
            output = subprocess.check_output(
                command, stderr=subprocess.DEVNULL, env=env
            )

        if format == json_format:
            parsed = BytesHeaderParser().parsebytes(output)
            try:
                remove_color_codes = re.compile(r"\x1b\[[0-9;]*m")
                parsed_as_string = re.sub(
                    remove_color_codes, "", parsed.as_string().strip()
                )
                return json.loads(parsed_as_string)
            except Exception:
                logger.debug(
                    f"JSON parse exception in decoding output from pip \
{action} {package}. Falling back to accumulating the byte stream",
                )
            accumulator = {}
            for key, value in parsed.items():
                accumulator[key] = value
            return accumulator
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
