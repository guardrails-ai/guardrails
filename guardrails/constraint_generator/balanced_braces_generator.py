from typing import Optional, Set

from guardrails.constraint_generator import ConstraintGenerator


class BalancedBracesGenerator(ConstraintGenerator):
    def __init__(self, max_depth: int = 1):
        self.max_depth = max_depth
        self.current_depth = 0

    def get_valid_tokens(self) -> Optional[Set[str]]:
        if self.current_depth < 0:
            return set()  # We have closed more than we opened.
        if self.current_depth >= self.max_depth:
            return {"}"}
        else:
            return {"{", "}"}

    def update_valid_tokens(self, token: str):
        net_ticks = token.count("{") - token.count("}")
        self.current_depth += net_ticks
