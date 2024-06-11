from abc import ABC, abstractmethod
from typing import Set, Optional


class ConstraintGenerator(ABC):
    @abstractmethod
    def get_valid_tokens(self) -> Optional[Set[str]]:
        """Given the current state of the constraint generator, return valid tokens.
        If there are no constraints on the tokens, will return None. If there are no
        valid tokens, will return an empty set.
        Will track the state of the constraint setup internally, but should be updated
        with update_valid_tokens."""
        ...

    @abstractmethod
    def update_valid_tokens(self, token: str): ...
