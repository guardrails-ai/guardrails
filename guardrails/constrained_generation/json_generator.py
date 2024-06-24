from collections import deque
from typing import List, Optional, Set

from guardrails.constrained_generation import ConstrainedGenerator
from guardrails.constrained_generation.number_generator import NumberGenerator
from guardrails.constrained_generation.union_generator import UnionGenerator


class JSONGenerator(ConstrainedGenerator):
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


def _make_any() -> ConstrainedGenerator:
    return UnionGenerator(
        QuotedStringGenerator(),
        NumberGenerator(is_integer=False),
        UnionGenerator(
            KeywordGenerator("true"),
            KeywordGenerator("false"),
            KeywordGenerator("null"),
        ),
        JSONObjectGenerator(),
        ArrayGenerator(),
    )


class JSONObjectGenerator(ConstrainedGenerator):
    def __init__(
            self,
            required_fields: Optional[List[str]] = None,
            required_types: Optional[List[str]] = None
    ):
        self.required_fields = required_fields
        self.required_types = required_types
        self.accumulator = ""
        self.pending_fields = list()
        self.active_subconstraint = None  # When we're making a child value...
        self.object_opened = False
        self.object_closed = False
        self.valid = True

    def get_valid_tokens(self) -> Optional[Set[str]]:
        if not self.valid:
            return set()  # Nothing to do.
        elif not self.object_opened:
            return {'{'}
        elif self.active_subconstraint is None:
            if not self.object_closed:
                return {'}'}
            else:
                return set()  # Nothing left to do.
        elif self.active_subconstraint.is_complete():
            # We finished as much as {"foo":"bar"
            # Now we need either a comma or an end parenthesis.
            return {',', '}'}
        else:
            return self.active_subconstraint.get_valid_tokens()

    def update_valid_tokens(self, token: str):
        for t in token:
            if not self.object_opened:
                if t == '{':
                    self.object_opened = True
                else:
                    self.valid = False
                    # Pick the new subconstraint.
                    # If we have subtypes or required fields, use those.

            if not self.is_complete() and self.valid:
                self.accumulator += t

    def is_complete(self) -> bool:
        return self.object_closed


class JSONValueGenerator(ConstrainedGenerator):
    """A JSON value is a `quoted_string colon (object|array|num|str|kw)`."""

    def __init__(self):
        self.finished_constraints = list()
        self.constraint_chain = [
            QuotedStringGenerator(),
            KeywordGenerator(":"),
            _make_any(),
        ]

    def get_valid_tokens(self) -> Optional[Set[str]]:
        if len(self.constraint_chain) == 0:
            return set()
        else:
            return self.constraint_chain[0].get_valid_tokens()

    def update_valid_tokens(self, token: str):
        for t in token:
            if len(self.constraint_chain) > 0:
                self.constraint_chain[0].update_valid_tokens(t)
                if self.constraint_chain[0].is_complete():
                    self.finished_constraints.append(self.constraint_chain[0])
                    self.constraint_chain = self.constraint_chain[1:]

    def get_name(self) -> str:
        """If the value is complete, return it without the double quotes."""
        assert len(self.finished_constraints) > 0
        return self.finished_constraints[0].accumulator[1:-1]

    def get_value(self):
        assert len(self.finished_constraints) > 2
        return self.finished_constraints[2].accumulator

    def is_complete(self) -> bool:
        return len(self.constraint_chain) == 0


class QuotedStringGenerator(ConstrainedGenerator):
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


class ArrayGenerator(ConstrainedGenerator):
    def __init__(self, array_type: Optional[str] = None, schema: Optional[dict] = None):
        super().__init__()
        self.base_schema = schema
        self.array_type = array_type
        self.is_opened = False
        self.is_closed = False
        self.had_initial_entry = False
        self.is_valid = True
        self.active_subconstraint = None
        self.accumulator = ""

    def get_valid_tokens(self) -> Optional[Set[str]]:
        if not self.is_opened:
            return {'['}
        elif self.active_subconstraint is not None:
            if self.active_subconstraint.is_complete():
                return {',', ']'}
            else:
                return self.active_subconstraint.get_valid_tokens()
        elif not self.is_closed:
            # If we have had a value already, like [1, ], then we can offer a comma.
            if self.had_initial_entry and self.active_subconstraint is None:
                return {',', ']'}
            else:
                return {']'}
        else:
            return set()  # We got into a bad state somehow.

    def update_valid_tokens(self, token: str):
        for t in token:
            if not self.is_opened:
                if t != '[':
                    self.is_valid = False
                else:
                    self.is_opened = True
                    # If we have type info, make the subconstraint this type.
                    # Else make it generic.
                    self.active_subconstraint = _make_any()
            elif self.active_subconstraint.is_complete():
                # Finished ["this"
                # Now we can either add a comma or close up.
                if t == ',':
                    self.active_subconstraint = _make_any()
                elif t == ']':
                    self.is_closed = True
                    self.is_valid = True
                else:
                    self.is_valid = False
            elif not self.active_subconstraint.is_complete():
                self.active_subconstraint.update_valid_tokens(t)

            if self.is_valid:
                self.accumulator += t

    def is_complete(self) -> bool:
        return self.is_closed


class KeywordGenerator(ConstrainedGenerator):
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
