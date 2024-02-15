import os
from dataclasses import dataclass
from os.path import expanduser
from typing import Optional

from guardrails.classes.generic.serializeable import Serializeable


@dataclass
class Credentials(Serializeable):
    id: Optional[str] = None
    client_id: Optional[str] = None
    client_secret: Optional[str] = None
    no_metrics: Optional[bool] = False

    @staticmethod
    def from_rc_file() -> "Credentials":
        try:
            home = expanduser("~")
            guardrails_rc = os.path.join(home, ".guardrailsrc")
            with open(guardrails_rc) as rc_file:
                lines = rc_file.readlines()
                creds = {}
                for line in lines:
                    key, value = line.split("=", 1)
                    creds[key.strip()] = value.strip()
                rc_file.close()
                return Credentials.from_dict(creds)

        except FileNotFoundError:
            return Credentials.from_dict({})  # type: ignore
