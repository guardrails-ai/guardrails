# Validation Outcome

## ValidationOutcome

#### raw\_llm\_output: `Optional[str]`

```python
raw_llm_output = Field(
    description="The raw, unchanged output from the LLM call.", default=None ...
```

The raw, unchanged output from the LLM call.

#### validated\_output: `Optional[OT]`

```python
validated_output = Field(
    description="The validated, and potentially fixed,"
    " output from  ...
```

The validated, and potentially fixed, output from the LLM call after
passing through validation.

#### reask: `Optional[ReAsk]`

```python
reask = Field(
    description="If validation continuously fails and all allocated"
    " reasks are ...
```

If validation continuously fails and all allocated reasks are used, this
field will contain the final reask that would have been sent to the LLM if
additional reasks were available.

#### validation\_passed: `bool`

```python
validation_passed = Field(
    description="A boolean to indicate whether or not"
    " the LLM outp ...
```

A boolean to indicate whether or not the LLM output passed validation.

If this is False, the validated_output may be invalid.

#### error: `Optional[str]`

```python
error = Field(default=None)
```

If the validation failed, this field will contain the error message.

#### from\_guard\_history

```python
@classmethod
def from_guard_history(cls, call: Call)
```

Create a ValidationOutcome from a history Call object.

#### \_\_iter\_\_

```python
def __iter__() -> Iterator[Union[Optional[str], Optional[OT], Optional[ReAsk],
                                 bool, Optional[str]]]
```

Iterate over the ValidationOutcome's fields.

#### \_\_getitem\_\_

```python
def __getitem__(keys)
```

Get a subset of the ValidationOutcome's fields.

