import os
import sys
from dataclasses import dataclass
from os.path import expanduser

from guardrails.cli_dir.server.serializeable import Serializeable


@dataclass
class Credentials(Serializeable):
    id: str
    client_id: str
    client_secret: str
    no_metrics: bool = False

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

        except FileNotFoundError as e:
            print(
                "Guardrails Hub credentials not found!\nSign up to use the Hub here: {insert url}"
            )
            sys.exit(1)
