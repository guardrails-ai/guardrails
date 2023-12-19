# Helper Classes
## Stack

Built-in mutable sequence.

If no argument is given, the constructor creates a new empty list.
The argument must be an iterable if specified.

### append `classmethod_descriptor`

```
append(
  self,
  object
)
```

Append object to the end of the list.

### at `classfunction`

```
at(
  self,
  index: int
) -> typing.Optional[~T]
```

Returns the item located at the index.

If the index does not exist in the stack (Overflow or
Underflow), None is returned instead.

### bottom `classproperty`

Returns the item on the bottom of the stack without removing it.

Same as Stack.first.

### clear `classmethod_descriptor`

```
clear(
  self
)
```

Remove all items from list.

### copy `classfunction`

```
copy(
  self
) -> Stack[T]
```

Returns a copy of the current Stack.

### count `classmethod_descriptor`

```
count(
  self,
  value
)
```

Return number of occurrences of value.

### empty `classfunction`

```
empty(
  self
) -> <class 'bool'>
```

Tests if this stack is empty.

### extend `classmethod_descriptor`

```
extend(
  self,
  iterable
)
```

Extend list by appending elements from the iterable.

### first `classproperty`

Returns the first item of the stack without removing it.

Same as Stack.bottom.

### index `classmethod_descriptor`

```
index(
  self,
  value,
  start=0,
  stop=9223372036854775807
)
```

Return first index of value.

Raises ValueError if the value is not present.

### insert `classmethod_descriptor`

```
insert(
  self,
  index,
  object
)
```

Insert object before index.

### last `classproperty`

Returns the last item of the stack without removing it.

Same as Stack.top.

### length `classproperty`

Returns the number of items in the Stack.

### peek `classfunction`

```
peek(
  self
) -> typing.Optional[~T]
```

Looks at the object at the top (last/most recently added) of this
stack without removing it from the stack.

### pop `classfunction`

```
pop(
  self
) -> typing.Optional[~T]
```

Removes the object at the top of this stack and returns that object
as the value of this function.

### push `classfunction`

```
push(
  self,
  item: ~T
) -> None
```

Pushes an item onto the top of this stack.

Proxy of List.append

### remove `classmethod_descriptor`

```
remove(
  self,
  value
)
```

Remove first occurrence of value.

Raises ValueError if the value is not present.

### reverse `classmethod_descriptor`

```
reverse(
  self
)
```

Reverse *IN PLACE*.

### search `classfunction`

```
search(
  self,
  x: ~T
) -> typing.Optional[int]
```

Returns the 0-based position of the last item whose value is equal
to x on this stack.

We deviate from the typical 1-based position used by Stack
classes (i.e. Java) because most python users (and developers in
general) are accustomed to 0-based indexing.

### sort `classmethod_descriptor`

```
sort(
  self,
  key=None,
  reverse=False
)
```

Sort the list in ascending order and return None.

The sort is in-place (i.e. the list itself is modified) and stable (i.e. the
order of two equal elements is maintained).

If a key function is given, apply it once to each list item and sort them,
ascending or descending, according to their function values.

The reverse flag can be set to sort in descending order.

### top `classproperty`

Returns the item on the top of the stack without removing it.

Same as Stack.last.


