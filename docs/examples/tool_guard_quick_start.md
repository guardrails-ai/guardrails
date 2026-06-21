# @guard_tool Quick Start Guide

Protect agent tools with input/output validation in just a few lines of code.

## Installation

The `@guard_tool` decorator is built into Guardrails AI:

```python
from guardrails import guard_tool, ToolGuard, ToolValidationError
from guardrails.types.on_fail import OnFailAction
```

## Minimal Example

```python
from guardrails import guard_tool
from guardrails.validators import ValidEmail

@guard_tool(
    input_validators={"email": [ValidEmail()]}
)
def send_notification(email: str) -> dict:
    """Send notification - validates email before execution."""
    return {"status": "sent"}

# Valid input - executes normally
send_notification("user@example.com")

# Invalid input - raises ToolValidationError by default
send_notification("not-an-email")
```

## Common Patterns

### 1. Validate Tool Arguments

```python
from guardrails import guard_tool
from guardrails.validators import ValidRange

@guard_tool(
    input_validators={"amount": [ValidRange(min=0, max=1000)]}
)
def transfer_money(amount: float) -> dict:
    """Transfer money with validated amount."""
    return {"status": "transferred"}
```

### 2. Validate Tool Outputs

```python
from guardrails import guard_tool
from guardrails.validators import ValidJSON

@guard_tool(
    output_validators=[ValidJSON()]
)
def fetch_user_data(user_id: int) -> dict:
    """Fetch user data - response validated as JSON."""
    return {"id": user_id, "name": "User"}
```

### 3. Both Input and Output Validation

```python
@guard_tool(
    input_validators={"user_id": [UserIDValidator()]},
    output_validators=[UserDataValidator()]
)
def get_user_profile(user_id: int) -> dict:
    """Get user profile with full validation."""
    return {"id": user_id}
```

### 4. Validate Specific Parameters

```python
@guard_tool(
    input_validators={
        "email": [EmailValidator()],
        "age": [ValidRange(min=18, max=120)],
        "country": [CountryValidator()]
    }
)
def register_user(email: str, age: int, country: str) -> dict:
    """Register user with per-parameter validation."""
    return {"status": "registered"}
```

## Error Handling Strategies

### EXCEPTION (Default) - Strict Validation

```python
@guard_tool(
    input_validators={"password": [PasswordValidator()]},
    on_input_fail=OnFailAction.EXCEPTION
)
def change_password(password: str) -> dict:
    """Strong password required - raises error on failure."""
    return {"status": "changed"}

try:
    change_password("weak")
except ToolValidationError as e:
    print(f"Invalid: {e.failures}")
```

### FIX - Auto-Correct

```python
@guard_tool(
    input_validators={"email": [EmailValidator()]},
    on_input_fail=OnFailAction.FIX
)
def subscribe(email: str) -> dict:
    """Email auto-corrected if invalid."""
    return {"status": "subscribed"}

subscribe("user@domain")  # Auto-fixed to "user@domain.com"
```

### FILTER - Remove Invalid

```python
@guard_tool(
    input_validators={"tags": [ValidTags()]},
    on_input_fail=OnFailAction.FILTER
)
def add_tags(tags: list) -> dict:
    """Invalid tags removed."""
    return {"tags": tags}

add_tags(["good", "bad!", "valid"])  # Returns ["good", "valid"]
```

### REFRAIN - Don't Execute

```python
@guard_tool(
    input_validators={"path": [SafePathValidator()]},
    on_input_fail=OnFailAction.REFRAIN
)
def read_file(path: str) -> str:
    """Unsafe paths block execution."""
    return open(path).read()

read_file("../../etc/passwd")  # Blocked - returns None
```

### NOOP - Warn But Execute

```python
@guard_tool(
    input_validators={"beta": [BetaFeatureValidator()]},
    on_input_fail=OnFailAction.NOOP
)
def experimental(beta: bool) -> dict:
    """Warning logged but tool executes."""
    return {"result": "experimental"}
```

## Tracking Execution

```python
from guardrails import ToolGuard, guard_tool

# Create manager
guard = ToolGuard(name="my_tools")

# Decorate tools
@guard_tool(
    input_validators={"id": [IDValidator()]},
    tool_guard=guard
)
def get_data(id: int) -> dict:
    return {"id": id, "data": "..."}

# Use tools
get_data(1)
get_data(2)
get_data(999)

# Check results
print(f"Total calls: {len(guard.execution_history)}")
print(f"Pass rate: {guard.get_statistics()['validation_pass_rate']}")

# View details
for result in guard.execution_history:
    print(f"{result.tool_name}: {result.validation_passed}")
```

## Async Support

```python
@guard_tool(
    input_validators={"url": [URLValidator()]}
)
async def fetch_url(url: str) -> str:
    """Async tool with validation."""
    import aiohttp
    async with aiohttp.ClientSession() as session:
        async with session.get(url) as resp:
            return await resp.text()

# Use in async context
import asyncio
asyncio.run(fetch_url("https://example.com"))
```

## Integration Examples

### LangGraph

```python
from langchain_core.tools import tool
from guardrails import guard_tool

@tool
@guard_tool(input_validators={"q": [QueryValidator()]})
def search(q: str) -> str:
    """Search the web."""
    return "results..."

# Use with LangGraph
from langgraph.graph import StateGraph
graph = StateGraph()
# ... configure with search tool
```

### CrewAI

```python
from crewai import Tool
from guardrails import guard_tool

@guard_tool(input_validators={"file": [FileValidator()]})
def read_file(file: str) -> str:
    """Read file content."""
    return open(file).read()

tool = Tool(
    name="read_file",
    func=read_file,
    description="Read file content"
)
```

### AutoGen

```python
from guardrails import guard_tool

@guard_tool(input_validators={"query": [QueryValidator()]})
def search_db(query: str) -> dict:
    """Search database."""
    return {"results": []}

agent.register_function(search_db)
```

## Cheatsheet

| Task | Code |
|------|------|
| Basic validation | `@guard_tool(input_validators={"param": [Validator()]})` |
| Output validation | `@guard_tool(output_validators=[Validator()])` |
| Handle failure | `on_input_fail=OnFailAction.FIX` |
| Track execution | `tool_guard=ToolGuard()` |
| Multiple validators | `{"param": [Validator1(), Validator2()]}` |
| Async tool | `async def tool(...):` |
| Get results | `guard.execution_history` |
| Get stats | `guard.get_statistics()` |
| Catch error | `except ToolValidationError as e:` |

## Next Steps

- Read the [full integration guide](./tool_guard_agent_integration.md)
- Check [built-in validators](../api_reference/validator.md)
- View [unit tests](../../tests/unit_tests/test_tool_guard.py) for more examples
- Configure [custom validators](../api_reference/validator.md) for your domain

## Support

For issues or questions:
- [GitHub Issues](https://github.com/guardrails-ai/guardrails/issues)
- [Documentation](https://docs.guardrailsai.com/)
- [Community Discord](https://discord.gg/guardrails)
