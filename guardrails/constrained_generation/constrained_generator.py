from abc import ABC, abstractmethod
from typing import Set, Optional


class ConstrainedGenerator(ABC):
    @abstractmethod
    def is_complete(self) -> bool:
        """Returns 'true' if the tokens that have been provided so far are sufficient to
        complete the given object.  For example, an integer constraint would not be
        complete after getting the "-" token, but would be complete after "-1". A
        balanced quote constraint might go from not complete to complete and back as
        more quotes are added and removed."""
        ...
        # JC: If a constraint is violated, is it complete?

    @abstractmethod
    def get_valid_tokens(self) -> Optional[Set[str]]:
        """Given the current state of the constraint generator, return valid tokens.
        If there are no constraints on the tokens, will return None. If there are no
        valid tokens, will return an empty set.
        Will track the state of the constraint setup internally, but should be updated
        with update_valid_tokens."""
        ...

    @abstractmethod
    def update_valid_tokens(self, token: str):
        """Update the internal state of the constraint generator. If the given token
        does not match with any of the current valid tokens then one can expect the
        next call to get_valid_tokens to return the empty set."""
        ...
