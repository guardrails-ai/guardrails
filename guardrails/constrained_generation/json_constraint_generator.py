import string
from collections import deque
from typing import List, Optional, Set

from guardrails.constrained_generation import ConstraintGenerator


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


class JSONObjectConstraint(ConstraintGenerator):
    def __init__(
            self,
            required_fields: Optional[List[str]] = None,
            #required_types: Optional[List[str]] = None
    ):
        self.required_fields = required_fields
        #self.required_types = required_types
        self.accumulator = ""
        self.active_subconstraint = None  # When we're making a child value...
        self.object_opened = False
        self.object_closed = False
        self.valid = False

    def get_valid_tokens(self) -> Optional[Set[str]]:
        if not self.object_opened:
            return {'{'}
        elif self.active_subconstraint is None:
            if not self.object_closed:
                return {'}'}
            else:
                return set()  # Nothing left to do.

    def update_valid_tokens(self, token: str):
        pass

    def is_complete(self) -> bool:
        return self.object_closed


class JSONValueConstraint(ConstraintGenerator):
    """A JSON value is a `quoted_string colon (object|array|num|str|kw)`."""

    def __init__(self):
        self.accumulator = ""
        self.constraint_chain = [
            QuotedStringConstraintGenerator(),
            KeywordConstraintGenerator(":"),
            UnionConstraintGenerator(
                QuotedStringConstraintGenerator(),
                NumberConstraintGenerator(is_integer=False),
                UnionConstraintGenerator(
                    KeywordConstraintGenerator("true"),
                    KeywordConstraintGenerator("false"),
                    KeywordConstraintGenerator("null"),
                ),
                #KeywordConstraintGenerator("{"),
            ),
        ]

    def get_valid_tokens(self) -> Optional[Set[str]]:
        if len(self.constraint_chain) == 0:
            return set()
        else:
            return self.constraint_chain[0].get_valid_tokens()

    def update_valid_tokens(self, token: str):
        self.accumulator += token
        for t in token:
            if len(self.constraint_chain) > 0:
                self.constraint_chain[0].update_valid_tokens(t)
                if self.constraint_chain[0].is_complete():
                    self.constraint_chain = self.constraint_chain[1:]

    def is_complete(self) -> bool:
        return len(self.constraint_chain) == 0


class QuotedStringConstraintGenerator(ConstraintGenerator):
    """Accepts a string, starting with a double quote and ending with a double quote."""

    def __init__(self):
        self.accumulator = ""
        self.escape_active = False
        self.quote_active = False

    def get_valid_tokens(self) -> Optional[Set[str]]:
        if not self.accumulator:
            return {'"'}
        elif self.escape_active:
            return {'"', "\\", "b", "n", "t"}
        else:
            return None  # No constraints

    def update_valid_tokens(self, token: str):
        for t in token:
            self.accumulator += t
            if self.escape_active:
                self.escape_active = False
            elif t == "\\":
                self.escape_active = True
            elif t == '"':
                self.quote_active = not self.quote_active

    def is_complete(self) -> bool:
        return not self.quote_active and len(self.accumulator) > 2


class ArrayConstraintGenerator(JSONConstraintGenerator):
    def __init__(self, base_schema: dict, array_type: str, schema: dict):
        super().__init__(schema)
        self.base_schema = base_schema
        self.array_type = array_type
        self.is_opened = False
        self.is_closed = False
        self.is_valid = False
        self.active_subconstraint = None
        self.accumulator = ""

    def get_valid_tokens(self) -> Optional[Set[str]]:
        raise NotImplementedError("TODO!")

    def update_valid_tokens(self, token: str):
        raise NotImplementedError("TODO!")


class UnionConstraintGenerator(ConstraintGenerator):
    def __init__(self, *args):
        self.sub_constraints = list()
        for arg in args:
            assert isinstance(arg, ConstraintGenerator)
            self.sub_constraints.append(arg)

    def get_valid_tokens(self) -> Optional[Set[str]]:
        valid_tokens = set()
        for c in self.sub_constraints:
            new_valid_tokens = c.get_valid_tokens()
            if new_valid_tokens is None:
                return None  # No constraints!
            valid_tokens |= new_valid_tokens
        return valid_tokens

    def update_valid_tokens(self, token: str):
        for c in self.sub_constraints:
            c.update_valid_tokens(token)

    def is_complete(self) -> bool:
        return any([c.is_complete() for c in self.sub_constraints])


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
