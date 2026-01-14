from typing import List

from guardrails import Guard
from guardrails_api.classes.http_error import HttpError
from guardrails_api.clients.guard_client import GuardClient


class MemoryGuardClient(GuardClient):
    # key value pair of guard_name to guard
    guards = {}

    def __init__(self):
        self.initialized = True

    def get_guard(self, guard_name: str, as_of_date: str = None) -> Guard:
        guard = self.guards.get(guard_name, None)
        return guard

    def get_guards(self) -> List[Guard]:
        return list(self.guards.values())

    def create_guard(self, guard: Guard) -> Guard:
        self.guards[guard.name] = guard
        return guard

    def update_guard(self, guard_name: str, new_guard: Guard) -> Guard:
        old_guard = self.get_guard(guard_name)
        if old_guard is None:
            raise HttpError(
                status=404,
                message="NotFound",
                cause="A Guard with the name {guard_name} does not exist!".format(
                    guard_name=guard_name
                ),
            )
        self.guards[guard_name] = new_guard
        return new_guard

    def upsert_guard(self, guard_name: str, new_guard: Guard) -> Guard:
        self.create_guard(new_guard)
        return new_guard

    def delete_guard(self, guard_name: str) -> Guard:
        deleted_guard = self.get_guard(guard_name)
        if deleted_guard is None:
            raise HttpError(
                status=404,
                message="NotFound",
                cause="A Guard with the name {guard_name} does not exist!".format(
                    guard_name=guard_name
                ),
            )
        del self.guards[guard_name]
        return deleted_guard
