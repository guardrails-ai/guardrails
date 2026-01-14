from typing import Union

import pytest

from guardrails.classes.generic.stack import Stack


@pytest.mark.parametrize("stack,expected_value", [(Stack(1, 2, 3), False), (Stack(), True)])
def test_empty(stack: Stack, expected_value: bool):
    result = stack.empty()

    assert result == expected_value


@pytest.mark.parametrize("stack,expected_value", [(Stack(1, 2, 3), 3), (Stack(), None)])
def test_peek(stack: Stack, expected_value: Union[int, None]):
    elem = stack.peek()

    assert elem == expected_value


@pytest.mark.parametrize(
    "stack,expected_value,expected_stack",
    [(Stack(1, 2, 3), 3, Stack(1, 2)), (Stack(), None, Stack())],
)
def test_pop(stack: Stack, expected_value: Union[int, None], expected_stack: Stack):
    elem = stack.pop()

    assert elem == expected_value
    assert stack == expected_stack


def test_push():
    stack = Stack()

    stack.push(1)
    assert stack[0] == 1
    assert stack[-1] == 1

    stack.push(2)
    assert stack[0] == 1
    assert stack[-1] == 2


@pytest.mark.parametrize("search_value,expected_index", [(1, 0), (2, 4), (3, 5), (5, None)])
def test_search(search_value: int, expected_index: int):
    stack = Stack(1, 2, 3, 4, 2, 3)

    index = stack.search(search_value)

    assert index == expected_index


@pytest.mark.parametrize("index,expected_value", [(1, 2), (3, None)])
def test_at(index: int, expected_value: int):
    stack = Stack(1, 2, 3)
    elem = stack.at(index)

    assert elem == expected_value


@pytest.mark.parametrize("stack,expected_value", [(Stack(1, 2, 3), 1), (Stack(), None)])
def test_first(stack: Stack, expected_value: int):
    elem = stack.first

    assert elem == expected_value


@pytest.mark.parametrize("stack,expected_value", [(Stack(1, 2, 3), 3), (Stack(), None)])
def test_last(stack: Stack, expected_value: int):
    elem = stack.last

    assert elem == expected_value


@pytest.mark.parametrize("stack,expected_value", [(Stack(1, 2, 3), 1), (Stack(), None)])
def test_bottom(stack: Stack, expected_value: int):
    elem = stack.bottom

    assert elem == expected_value


@pytest.mark.parametrize("stack,expected_value", [(Stack(1, 2, 3), 3), (Stack(), None)])
def test_top(stack: Stack, expected_value: int):
    elem = stack.top

    assert elem == expected_value


def test_length():
    stack = Stack()

    assert stack.length == 0

    stack.push(1)
    assert stack.length == 1
