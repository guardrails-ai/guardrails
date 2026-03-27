import json
import os
import re
import subprocess
import sys

from typing import Literal
import logging

from email.parser import BytesHeaderParser
from typing import List, Union


json_format: Literal["json"] = "json"
string_format: Literal["string"] = "string"

logger = logging.getLogger(__name__)

json_format = "json"
string_format = "string"


class PipProcessError(Exception):
    action: str
    package: str
    stderr: str = ""
    stdout: str = ""
    returncode: int = 1

    def __init__(
        self,
        action: str,
        package: str,
        stderr: str = "",
        stdout: str = "",
        returncode: int = 1,
    ):
        self.action = action
        self.package = package
        self.stderr = stderr
        self.stdout = stdout
        self.returncode = returncode
        message = (
            f"PipProcessError: {action} on '{package}' failed with"
            "return code {returncode}.\n"
            f"Stdout:\n{stdout}\n"
            f"Stderr:\n{stderr}"
        )
        super().__init__(message)


def pip_process_with_custom_exception(
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

        result = subprocess.run(
            command,
            env=env,
            capture_output=True,  # Capture both stdout and stderr
            text=True,  # Automatically decode to strings
            check=True,  # Automatically raise error on non-zero exit code
        )

        if format == json_format:
            try:
                remove_color_codes = re.compile(r"\x1b\[[0-9;]*m")
                parsed_as_string = re.sub(remove_color_codes, "", result.stdout.strip())
                return json.loads(parsed_as_string)
            except Exception:
                logger.debug(
                    f"JSON parse exception in decoding output from pip {action}"
                    f" {package}. Falling back to accumulating the byte stream",
                )
            accumulator = {}
            parsed = BytesHeaderParser().parsebytes(result.stdout.encode())
            for key, value in parsed.items():
                accumulator[key] = value
            return accumulator

        return result.stdout

    except subprocess.CalledProcessError as exc:
        raise PipProcessError(action, package, exc.stderr, exc.stdout, exc.returncode)
    except Exception as e:
        raise PipProcessError(action, package, stderr=str(e), stdout="", returncode=1)


def installer_process(
    action: str,
    package: str = "",
    flags: List[str] = [],
    format: Union[Literal["string"], Literal["json"]] = string_format,
    quiet: bool = False,
    no_color: bool = False,
    installer: str = "pip",
) -> Union[str, dict]:
    """Run a package install action using the specified installer (uv or pip).

    Args:
        action: The pip action to run (e.g., "install").
        package: The package name to act on.
        flags: Additional flags to pass to the installer.
        format: Output format ("string" or "json").
        quiet: Whether to suppress output.
        no_color: Whether to disable color output.
        installer: Package installer to use ("uv" or "pip").
    """
    try:
        if installer == "uv":
            command = ["uv", "pip", action]
        else:
            command = [sys.executable, "-m", "pip", action]

        if not quiet:
            installer_label = "uv pip" if installer == "uv" else "pip"
            logger.debug(
                f"running {installer_label} {action} {' '.join(flags)} {package}"
            )

        command.extend(flags)
        if package:
            command.append(package)

        env = dict(os.environ)
        if no_color:
            env["NO_COLOR"] = "true"

        result = subprocess.run(
            command,
            env=env,
            capture_output=True,
            text=True,
            check=True,
        )

        if format == json_format:
            try:
                remove_color_codes = re.compile(r"\x1b\[[0-9;]*m")
                parsed_as_string = re.sub(remove_color_codes, "", result.stdout.strip())
                return json.loads(parsed_as_string)
            except Exception:
                logger.debug(
                    f"JSON parse exception in decoding output from {action}"
                    f" {package}. Falling back to accumulating the byte stream",
                )
            accumulator = {}
            parsed = BytesHeaderParser().parsebytes(result.stdout.encode())
            for key, value in parsed.items():
                accumulator[key] = value
            return accumulator

        return result.stdout

    except subprocess.CalledProcessError as exc:
        raise PipProcessError(
            action, package, exc.stderr or "", exc.stdout or "", exc.returncode
        )
    except Exception as e:
        raise PipProcessError(action, package, stderr=str(e), stdout="", returncode=1)


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

        result = subprocess.run(
            command,
            env=env,
            capture_output=True,  # Capture both stdout and stderr
            text=True,  # Automatically decode to strings
            check=True,  # Automatically raise error on non-zero exit code
        )

        if format == json_format:
            try:
                remove_color_codes = re.compile(r"\x1b\[[0-9;]*m")
                parsed_as_string = re.sub(remove_color_codes, "", result.stdout.strip())
                return json.loads(parsed_as_string)
            except Exception:
                logger.debug(
                    f"JSON parse exception in decoding output from pip {action}"
                    f" {package}. Falling back to accumulating the byte stream",
                )
            accumulator = {}
            parsed = BytesHeaderParser().parsebytes(result.stdout.encode())
            for key, value in parsed.items():
                accumulator[key] = value
            return accumulator

        return result.stdout

    except subprocess.CalledProcessError as exc:
        logger.error(
            (
                f"Failed to {action} {package}\n"
                f"Exit code: {exc.returncode}\n"
                f"stderr: {(exc.stderr or '').strip()}\n"
                f"stdout: {(exc.stdout or '').strip()}"
            )
        )
        sys.exit(1)
    except Exception as e:
        logger.error(
            f"An unexpected exception occurred while trying to {action} {package}!",
            e,
        )
        sys.exit(1)
