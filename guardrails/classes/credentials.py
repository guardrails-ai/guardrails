import logging
import os
from dataclasses import dataclass
from os.path import expanduser
from typing import Optional

from guardrails.classes.generic.serializeable import Serializeable

BOOL_CONFIGS = set(["no_metrics", "enable_metrics", "use_remote_inferencing"])


@dataclass
class Credentials(Serializeable):
    id: Optional[str] = None
    token: Optional[str] = None
    no_metrics: Optional[bool] = False
    enable_metrics: Optional[bool] = True
    use_remote_inferencing: Optional[bool] = True

    @staticmethod
    def _to_bool(value: str) -> Optional[bool]:
        if value.lower() == "true":
            return True
        if value.lower() == "false":
            return False
        return None

    @staticmethod
    def has_rc_file() -> bool:
        home = expanduser("~")
        guardrails_rc = os.path.join(home, ".guardrailsrc")
        return os.path.exists(guardrails_rc)

    @staticmethod
    def from_rc_file(logger: Optional[logging.Logger] = None) -> "Credentials":
        try:
            if not logger:
                logger = logging.getLogger()
            home = expanduser("~")
            guardrails_rc = os.path.join(home, ".guardrailsrc")
            with open(guardrails_rc, encoding="utf-8") as rc_file:
                lines = rc_file.readlines()
                filtered_lines = list(filter(lambda l: l.strip(), lines))
                creds = {}
                for line in filtered_lines:
                    line_content = line.split("=", 1)
                    if len(line_content) != 2:
                        logger.warning(
                            """
                            Invalid line found in .guardrailsrc file!
                            All lines in this file should follow the format: key=value
                            Ignoring line contents...
                            """
                        )
                        logger.debug(f".guardrailsrc file location: {guardrails_rc}")
                    else:
                        key, value = line_content
                        key = key.strip()
                        value = value.strip()
                        if key in BOOL_CONFIGS:
                            value = Credentials._to_bool(value)

                        creds[key] = value

                rc_file.close()

                # backfill no_metrics, handle defaults
                # remove in 0.5.0
                no_metrics_val = creds.pop("no_metrics", None)
                if no_metrics_val is not None and creds.get("enable_metrics") is None:
                    creds["enable_metrics"] = not no_metrics_val

                creds_dict = Credentials.from_dict(creds)
                return creds_dict

        except FileNotFoundError:
            return Credentials.from_dict({})  # type: ignore
