import logging
import os
from dataclasses import dataclass
from os.path import expanduser
from typing import Optional

from guardrails.classes.generic.serializeable import Serializeable


@dataclass
class Credentials(Serializeable):
    id: Optional[str] = None
    token: Optional[str] = None
    no_metrics: Optional[bool] = False

    @staticmethod
    def from_rc_file(logger: Optional[logging.Logger] = None) -> "Credentials":
        try:
            if not logger:
                logger = logging.getLogger()
            home = expanduser("~")
            guardrails_rc = os.path.join(home, ".guardrailsrc")
            with open(guardrails_rc) as rc_file:
                lines = rc_file.readlines()
                filtered_lines = list(filter(lambda l: l.strip(), lines))
                creds = {}
                for line in filtered_lines:
                    line_content = line.split("=", 1)
                    if len(line_content) != 2:
                        logger.warn(
                            """
                            Invalid line found in .guardrailsrc file!
                            All lines in this file should follow the format: key=value
                            Ignoring line contents...
                            """
                        )
                        logger.debug(f".guardrailsrc file location: {guardrails_rc}")
                    else:
                        key, value = line_content
                        creds[key.strip()] = value.strip()
                rc_file.close()
                return Credentials.from_dict(creds)

        except FileNotFoundError:
            return Credentials.from_dict({})  # type: ignore
