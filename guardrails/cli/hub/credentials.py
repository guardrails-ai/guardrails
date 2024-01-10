import os
import sys

from dataclasses import dataclass
from guardrails.cli.logger import logger
from os.path import expanduser
from typing import Dict

from guardrails.cli.server.serializeable import Serializeable


@dataclass
class Credentials(Serializeable):
    client_id: str
    client_secret: str
    no_metrics: bool = False
    
    
    @staticmethod
    def from_rc_file() -> "Credentials":
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