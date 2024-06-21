from typing import Optional, Set

from guardrails.constrained_generation import ConstrainedGenerator


class UnionGenerator(ConstrainedGenerator):
    def __init__(self, *args):
        self.sub_constraints = list()
        for arg in args:
            assert isinstance(arg, ConstrainedGenerator)
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
