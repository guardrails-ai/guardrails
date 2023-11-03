from typing import Optional, TypeVar, cast

T = TypeVar("T")


class ListPlusPlus(list[T]):
    def __init__(self, *args):
        list.__init__(self, args)

    def at(self, index: int) -> Optional[T]:
        try:
            value = self[index]
            return value
        except IndexError:
            pass
