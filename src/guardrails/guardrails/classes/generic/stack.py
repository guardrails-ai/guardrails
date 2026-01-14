from typing import List, Optional, TypeVar

T = TypeVar("T")


class Stack(List[T]):
    def __init__(self, *args):
        super().__init__(args)

    def empty(self) -> bool:
        """Tests if this stack is empty."""
        return len(self) == 0

    def peek(self) -> Optional[T]:
        """Looks at the object at the top (last/most recently added) of this
        stack without removing it from the stack."""
        return self.at(-1)

    def pop(self) -> Optional[T]:
        """Removes the object at the top of this stack and returns that object
        as the value of this function."""
        try:
            value = super().pop()
            return value
        except IndexError:
            pass

    def push(self, item: T) -> None:
        """Pushes an item onto the top of this stack.

        Proxy of List.append
        """
        self.append(item)

    def search(self, x: T) -> Optional[int]:
        """Returns the 0-based position of the last item whose value is equal
        to x on this stack.

        We deviate from the typical 1-based position used by Stack
        classes (i.e. Java) because most python users (and developers in
        general) are accustomed to 0-based indexing.
        """
        copy = self.copy()
        copy.reverse()

        try:
            found_index = copy.index(x)
            return len(self) - found_index - 1
        except ValueError:
            pass

    def at(self, index: int, default: Optional[T] = None) -> Optional[T]:
        """Returns the item located at the index.

        If the index does not exist in the stack (Overflow or
        Underflow), None is returned instead.
        """
        try:
            value = self[index]
            return value
        except IndexError:
            return default

    def copy(self) -> "Stack[T]":
        """Returns a copy of the current Stack."""
        copy = super().copy()
        return Stack(*copy)

    @property
    def first(self) -> Optional[T]:
        """Returns the first item of the stack without removing it.

        Same as Stack.bottom.
        """
        return self.at(0)

    @property
    def last(self) -> Optional[T]:
        """Returns the last item of the stack without removing it.

        Same as Stack.top.
        """
        return self.at(-1)

    @property
    def bottom(self) -> Optional[T]:
        """Returns the item on the bottom of the stack without removing it.

        Same as Stack.first.
        """
        return self.at(0)

    @property
    def top(self) -> Optional[T]:
        """Returns the item on the top of the stack without removing it.

        Same as Stack.last.
        """
        return self.at(-1)

    @property
    def length(self) -> int:
        """Returns the number of items in the Stack."""
        return len(self)
