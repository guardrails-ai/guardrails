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
