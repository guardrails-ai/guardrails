# @guard_tool Decorator Reference

Input and output validation for autonomous agent tools.

## Import

```python
from guardrails import guard_tool, ToolGuard, ToolValidationError, ToolValidationResult
from guardrails.types.on_fail import OnFailAction
```

## Signature

```python
@guard_tool(
    input_validators: Optional[Union[List[Validator], Dict[str, List[Validator]]]] = None,
    output_validators: Optional[List[Validator]] = None,
    metadata: Optional[Dict[str, Any]] = None,
    tool_guard: Optional[ToolGuard] = None,
    on_input_fail: Optional[Union[str, OnFailAction]] = None,
    on_output_fail: Optional[Union[str, OnFailAction]] = None
)
def my_tool(...) -> Any:
    ...
```

## Parameters

### input_validators
**Type**: `Optional[Union[List[Validator], Dict[str, List[Validator]]]]`  
**Default**: `None`

Validators for tool input parameters.

- **List format**: Validators applied to entire arguments dict
  ```python
  input_validators=[CrossParameterValidator()]
  ```
- **Dict format**: Per-parameter validators
  ```python
  input_validators={
      "email": [EmailValidator()],
      "age": [ValidRange(0, 150)]
  }
  ```

### output_validators
**Type**: `Optional[List[Validator]]`  
**Default**: `None`

Validators for tool output.

```python
output_validators=[JSONValidator(), ResponseLengthValidator()]
```

### metadata
**Type**: `Optional[Dict[str, Any]]`  
**Default**: `None`

Additional context passed to validators.

```python
metadata={"context": "payment", "source": "external"}
```

### tool_guard
**Type**: `Optional[ToolGuard]`  
**Default**: `None`

ToolGuard manager for execution tracking.

```python
guard = ToolGuard(name="agent_tools")
@guard_tool(..., tool_guard=guard)
```

### on_input_fail
**Type**: `Optional[Union[str, OnFailAction]]`  
**Default**: `OnFailAction.EXCEPTION`

Action when input validation fails.

- `"exception"` or `OnFailAction.EXCEPTION`: Raise ToolValidationError
- `"fix"` or `OnFailAction.FIX`: Use validator's fix_value
- `"filter"` or `OnFailAction.FILTER`: Remove invalid elements
- `"refrain"` or `OnFailAction.REFRAIN`: Skip execution
- `"noop"` or `OnFailAction.NOOP`: Proceed despite failure

### on_output_fail
**Type**: `Optional[Union[str, OnFailAction]]`  
**Default**: `OnFailAction.EXCEPTION`

Action when output validation fails. Same options as `on_input_fail`.

## Returns

The decorated function with validation wrapper applied. Function signature unchanged.

## Raises

### ToolValidationError

Raised when validation fails with `on_*_fail=OnFailAction.EXCEPTION`.

```python
try:
    tool(invalid_input)
except ToolValidationError as e:
    print(e.failures)  # Dict[str, List[str]]
```

## Example

```python
from guardrails import guard_tool, Validator, ToolGuard
from guardrails.validators import EmailValidator, ValidRange
from guardrails.types.on_fail import OnFailAction

# Create manager for tracking
guard = ToolGuard(name="api_tools")

@guard_tool(
    input_validators={
        "email": [EmailValidator()],
        "age": [ValidRange(min=18, max=120)]
    },
    output_validators=[JSONValidator()],
    on_input_fail=OnFailAction.FIX,
    on_output_fail=OnFailAction.EXCEPTION,
    tool_guard=guard,
    metadata={"service": "registration"}
)
def register_user(email: str, age: int) -> dict:
    """Register user with validated inputs and outputs."""
    return {
        "id": 123,
        "email": email,
        "age": age,
        "status": "active"
    }

# Use
result = register_user("user@example.com", 25)

# Check execution
history = guard.execution_history
stats = guard.get_statistics()
```

## Supported Patterns

### Async Functions

```python
@guard_tool(...)
async def async_tool(param: str) -> dict:
    import aiohttp
    async with aiohttp.ClientSession() as session:
        return await fetch(session, param)

# Use in async context
await async_tool("value")
```

### Multiple Validators

```python
@guard_tool(
    input_validators={
        "password": [
            LengthValidator(min=8),
            ComplexityValidator(),
            HistoryValidator()
        ]
    }
)
def set_password(password: str) -> dict:
    ...
```

### Default Parameters

```python
@guard_tool(input_validators={"required": [Validator()]})
def tool(required: str, optional: str = "default"):
    ...

# Optional param uses default if not provided
tool("value")
```

### Variadic Arguments

```python
@guard_tool(input_validators={"first": [Validator()]})
def tool(first: str, *args, **kwargs):
    ...

tool("value", extra1, extra2, key="value")
```

## ToolGuard Manager

```python
guard = ToolGuard(name="my_tools")

# Register manually (automatic with @guard_tool)
tool_id = guard.register_tool(
    tool_name="my_tool",
    input_validators=[...],
    output_validators=[...]
)

# Get history
all_history = guard.execution_history
tool_history = guard.get_execution_history("tool_name")

# Get statistics
stats = guard.get_statistics()
tool_stats = guard.get_statistics("tool_name")
```

## ToolValidationResult

Individual execution result.

```python
result = guard.execution_history[0]

result.tool_name                 # str
result.validation_passed         # bool
result.input_validation_passed   # bool
result.output_validation_passed  # bool
result.input_failures            # Dict[str, Any]
result.output_failures           # Dict[str, Any]
result.validated_inputs          # Dict[str, Any]
result.validated_output          # Any
result.execution_time_ms         # float
result.input_on_fail_action      # OnFailAction
result.output_on_fail_action     # OnFailAction

result_dict = result.to_dict()   # Serialize
```

## Error Handling

### Validation Error

```python
from guardrails import ToolValidationError

try:
    tool(invalid_args)
except ToolValidationError as e:
    # e.failures: Dict[str, List[str]] - errors by parameter
    for param, errors in e.failures.items():
        print(f"{param}: {errors}")
```

### Tool Execution Error

Tool exceptions during execution are not caught:

```python
@guard_tool(input_validators=...)
def tool(x: int):
    return 1 / x  # Raises ZeroDivisionError if x=0

# Input validation passes if x=0, but tool raises ZeroDivisionError
try:
    tool(0)
except ZeroDivisionError:
    # Exception propagates normally
    pass
```

## Configuration

### Disable Tracing

```python
import os
os.environ["GUARDRAILS_DISABLE_TRACING"] = "true"

# Or via settings
from guardrails.settings import settings
settings.disable_tracing = True
```

## Integration Examples

### LangChain Tool

```python
from langchain_core.tools import tool

@tool
@guard_tool(input_validators={"q": [QueryValidator()]})
def search(q: str) -> str:
    """Search."""
    return "results"
```

### CrewAI

```python
from crewai import Tool

@guard_tool(input_validators=...)
def my_func(param: str) -> dict:
    ...

tool = Tool(
    name="my_tool",
    func=my_func,
    description="Tool description"
)
```

### AutoGen

```python
agent.register_function(
    @guard_tool(input_validators=...)
    def my_tool(...):
        ...
)
```

## See Also

- [Tool Guard Integration Guide](./tool_guard_agent_integration.md)
- [Tool Guard Quick Start](./tool_guard_quick_start.md)
- [Tool Guard Architecture](./tool_guard_architecture.md)
- [Validator Reference](./validator.md)
- [OnFailAction Reference](./on_fail.md)
