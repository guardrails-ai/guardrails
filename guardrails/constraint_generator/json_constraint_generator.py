import string
from collections import deque
from typing import Optional, Set

from guardrails.constraint_generator import ConstraintGenerator


class JSONConstraintGenerator(ConstraintGenerator):
    def __init__(self, schema: dict):
        self.accumulator = ""
        self.schema = schema
        self.state_stack = deque()

    def get_valid_tokens(self) -> Optional[Set[str]]:
        return {"a"}

    def update_valid_tokens(self, token: str):
        pass

    def is_complete(self) -> bool:
        """Returns 'true' if the given object is complete now."""
        return False


class JSONValueConstraint(ConstraintGenerator):
    """A JSON value is a `quoted string colon (object | array | number | string | kw)"""

    def __init__(self):
        self.accumulator = ""
        self.constraint_chain = [
            QuotedStringConstraintGenerator(),
            KeywordConstraintGenerator(":"),
            UnionConstraintGenerator(),
        ]

    def get_valid_tokens(self) -> Optional[Set[str]]:
        pass

    def update_valid_tokens(self, token: str):
        pass

    def is_complete(self) -> bool:
        pass


class QuotedStringConstraintGenerator(ConstraintGenerator):
    """Accepts a string, starting with a double quote and ending with a double quote."""

    def __init__(self):
        self.accumulator = ""
        self.escape_active = False

    def get_valid_tokens(self) -> Optional[Set[str]]:
        if not self.accumulator:  # Empty
            return {'"'}
        else:
            pass

    def update_valid_tokens(self, token: str):
        pass

    def is_complete(self) -> bool:
        return False


class ArrayConstraintGenerator(JSONConstraintGenerator):
    def __init__(self, base_schema: dict, array_type: str, schema: dict):
        super().__init__(schema)
        self.base_schema = base_schema
        self.array_type = array_type
        self.data = list()

    def get_valid_tokens(self) -> Optional[Set[str]]:
        pass

    def update_valid_tokens(self, token: str):
        pass


class UnionConstraintGenerator(ConstraintGenerator):
    def __init__(self, a: ConstraintGenerator, b: ConstraintGenerator):
        self.a = a
        self.b = b

    def get_valid_tokens(self) -> Optional[Set[str]]:
        return self.a.get_valid_tokens() | self.b.get_valid_tokens()

    def update_valid_tokens(self, token: str):
        self.a.update_valid_tokens(token)
        self.b.update_valid_tokens(token)

    def is_complete(self) -> bool:
        return self.a.is_complete() or self.b.is_complete()


class KeywordConstraintGenerator(ConstraintGenerator):
    """This might not seem like the most useful thing in the world, but it helps keep
    our model on the rails if we need to generate something like 'false' or 'true'."""

    def __init__(self, keyword: str, token_length_cap: int = 1):
        """Constrains output to the given keyword.  If the token_length_cap is set,
        will produce all sub-keywords up to the given length as valid tokens, i.e.
        keyword=foobezbar -> {'f', 'fo', 'foo', 'foob', ...}."""
        self.token_length_cap = token_length_cap
        self.keyword = keyword
        self.violated = False

    def is_complete(self) -> bool:
        return len(self.keyword) == 0 and not self.violated

    def get_valid_tokens(self) -> Optional[Set[str]]:
        if self.violated or self.is_complete():
            return set()
        valid = set()
        for i in range(1, self.token_length_cap + 1):
            valid.add(self.keyword[:i])
        return valid

    def update_valid_tokens(self, token: str):
        if self.keyword.startswith(token):
            self.keyword = self.keyword[len(token) :]
        else:
            # TODO: Log an attempt to update with an invalid token?
            self.violated = True
            self.keyword = ""


class NumberConstraintGenerator(ConstraintGenerator):
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
