from typing import List, Union

from guardrails import Guard
from guardrails_api_client import Guard as GuardStruct


class GuardClient:
    def __init__(self):
        self.initialized = True

    def get_guard(self, guard_name: str, as_of_date: str = None) -> Union[GuardStruct, Guard]:
        raise NotImplementedError

    def get_guards(self) -> List[Union[GuardStruct, Guard]]:
        raise NotImplementedError

    def create_guard(self, guard: Union[GuardStruct, Guard]) -> Union[GuardStruct, Guard]:
        raise NotImplementedError

    def update_guard(
        self, guard_name: str, guard: Union[GuardStruct, Guard]
    ) -> Union[GuardStruct, Guard]:
        raise NotImplementedError

    def upsert_guard(
        self, guard_name: str, guard: Union[GuardStruct, Guard]
    ) -> Union[GuardStruct, Guard]:
        raise NotImplementedError

    def delete_guard(self, guard_name: str) -> Union[GuardStruct, Guard]:
        raise NotImplementedError
