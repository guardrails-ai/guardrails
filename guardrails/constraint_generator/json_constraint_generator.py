from typing import Optional, Set

from guardrails.constraint_generator import ConstraintGenerator


class JsonConstraintGenerator(ConstraintGenerator):
    def get_valid_tokens(self) -> Optional[Set[str]]:
        return {"a"}

    def update_valid_tokens(self, token: str):
        pass
