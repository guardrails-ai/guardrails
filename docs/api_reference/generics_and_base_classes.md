# Generics And Base Classes

## ArbitraryModel

```python
class ArbitraryModel(BaseModel)
```

Empty Pydantic model with a config that allows arbitrary types.

## Stack

```python
class Stack(List[T])
```

#### empty

```python
def empty() -> bool
```

Tests if this stack is empty.

#### peek

```python
def peek() -> Optional[T]
```

Looks at the object at the top (last/most recently added) of this
stack without removing it from the stack.

#### pop

```python
def pop() -> Optional[T]
```

Removes the object at the top of this stack and returns that object
as the value of this function.

#### push

```python
def push(item: T) -> None
```

Pushes an item onto the top of this stack.

Proxy of List.append

Limits Stack Length to _max_length entries

#### search

```python
def search(x: T) -> Optional[int]
```

Returns the 0-based position of the last item whose value is equal
to x on this stack.

We deviate from the typical 1-based position used by Stack
classes (i.e. Java) because most python users (and developers in
general) are accustomed to 0-based indexing.

#### at

```python
def at(index: int, default: Optional[T] = None) -> Optional[T]
```

Returns the item located at the index.

If the index does not exist in the stack (Overflow or
Underflow), None is returned instead.

#### copy

```python
def copy() -> "Stack[T]"
```

Returns a copy of the current Stack.

#### first

```python
@property
def first() -> Optional[T]
```

Returns the first item of the stack without removing it.

Same as Stack.bottom.

#### last

```python
@property
def last() -> Optional[T]
```

Returns the last item of the stack without removing it.

Same as Stack.top.

#### bottom

```python
@property
def bottom() -> Optional[T]
```

Returns the item on the bottom of the stack without removing it.

Same as Stack.first.

#### top

```python
@property
def top() -> Optional[T]
```

Returns the item on the top of the stack without removing it.

Same as Stack.last.

#### length

```python
@property
def length() -> int
```

Returns the number of items in the Stack.

