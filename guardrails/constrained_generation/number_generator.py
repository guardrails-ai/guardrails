import string
from typing import Optional, Set

from guardrails.constrained_generation import ConstrainedGenerator


class NumberConstrainedGenerator(ConstrainedGenerator):
    def __init__(self, is_integer: bool, allow_leading_period: bool = False):
        super().__init__()
        self.accumulator = ""
        self.decimal_placed = False
        self.is_integer = is_integer
        self.allow_leading_period = allow_leading_period
        self.valid = True

    def is_complete(self) -> bool:
        if not self.valid:
            return False
        try:
            if self.is_integer:
                int(self.accumulator, 10)  # Force base-10.
            else:
                float(self.accumulator)
            return True
        except ValueError:
            return False

    def get_valid_tokens(self) -> Optional[Set[str]]:
        if not self.valid:
            return set()
        valid_tokens = {
            "0",
            "1",
            "2",
            "3",
            "4",
            "5",
            "6",
            "7",
            "8",
            "9",
        }
        if len(self.accumulator) == 0:
            valid_tokens.add("-")
        if not self.decimal_placed and not self.is_integer:
            # Can't start with '.' normally, so make sure we have at least one number.
            # Also make sure it's not just a '-' or '+'.
            if (
                self.allow_leading_period
                or len(self.accumulator) > 1
                or (len(self.accumulator) > 0 and self.accumulator[0] != "-")
            ):
                valid_tokens.add(".")
        return valid_tokens

    def update_valid_tokens(self, token: str):
        for t in token:
            self.valid = self.valid and any(
                [
                    t in string.digits,
                    (t == "-" and len(self.accumulator) == 0),
                    (
                        t == "."
                        and not self.decimal_placed
                        and len(self.accumulator) > 0
                    ),
                ]
            )
            self.accumulator += t
            if t == ".":
                self.decimal_placed = True
