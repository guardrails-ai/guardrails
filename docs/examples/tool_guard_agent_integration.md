# Tool Guard: Agent Tool Input/Output Validation

The `@guard_tool` decorator and `ToolGuard` manager enable secure tool execution in autonomous agent frameworks by validating tool inputs before execution and outputs before returning to the LLM context.

## Overview

### Key Features

- **Input Validation**: Validates LLM-generated tool arguments before execution (prevents injection attacks, out-of-bounds parameters)
- **Output Validation**: Validates returned data before sending back to the LLM (prevents prompt injection from poisoned sources)
- **OnFailAction Support**: Multiple failure handling strategies (EXCEPTION, FIX, FILTER, REFRAIN, NOOP)
- **OpenTelemetry Integration**: Full observability with OpenInference semantic conventions
- **Execution Tracking**: ToolGuard manager tracks all tool executions with validation history and statistics

### Addresses Security Concerns

- **OWASP LLM08 - Excessive Agency**: Validates tool arguments to prevent out-of-scope actions
- **OWASP LLM07 - Insecure Plugin Design**: Validates tool outputs before they reach the LLM

## Basic Usage

### Simple Decorator Pattern

```python
from guardrails import guard_tool, Validator
from guardrails.types.on_fail import OnFailAction

@guard_tool(
    input_validators={"email": [EmailValidator()]},
    output_validators=[SafeHTMLValidator()],
    on_input_fail=OnFailAction.EXCEPTION,
    on_output_fail=OnFailAction.FIX
)
def send_email(email: str, message: str) -> dict:
    """Send an email with validated inputs and outputs."""
    # Tool implementation
    return {"status": "sent", "email": email}
```

### With ToolGuard Manager

```python
from guardrails import ToolGuard, guard_tool

# Create a ToolGuard instance to track all tool executions
tool_guard = ToolGuard(name="my_agent_tools")

@guard_tool(
    input_validators={"user_id": [UserIDValidator()]},
    output_validators=[JSONValidator()],
    tool_guard=tool_guard
)
def fetch_user_data(user_id: int) -> dict:
    """Fetch user data with execution tracking."""
    # Tool implementation
    return {"id": user_id, "name": "User"}

# Check execution history
history = tool_guard.get_execution_history()
stats = tool_guard.get_statistics()
print(f"Success rate: {stats['validation_pass_rate']*100}%")
```

## Integration with Agent Frameworks

### LangGraph Integration

```python
from langchain_core.tools import tool
from guardrails import guard_tool, ToolGuard
from guardrails.validators import URLValidator

tool_guard = ToolGuard(name="langgraph_tools")

@tool
@guard_tool(
    input_validators={"url": [URLValidator()]},
    tool_guard=tool_guard,
    on_input_fail=OnFailAction.EXCEPTION
)
def fetch_webpage(url: str) -> str:
    """Fetch and return webpage content."""
    import requests
    response = requests.get(url)
    return response.text

# Use in LangGraph workflow
from langgraph.graph import StateGraph

tools = [fetch_webpage]

# ... rest of workflow configuration
```

### CrewAI Integration

```python
from crewai import Agent, Tool
from guardrails import guard_tool, ToolGuard
from guardrails.validators import ValidRange

tool_guard = ToolGuard(name="crewai_agent_tools")

@guard_tool(
    input_validators={
        "amount": [ValidRange(min=0, max=10000)],
        "account": [AccountIDValidator()]
    },
    on_input_fail=OnFailAction.FIX,
    tool_guard=tool_guard
)
def transfer_funds(account: str, amount: float) -> dict:
    """Transfer funds between accounts with validated inputs."""
    # Implementation with guardrail protection
    return {"status": "success", "amount": amount}

# Create Tool for CrewAI
transfer_tool = Tool(
    name="transfer_funds",
    func=transfer_funds,
    description="Transfer funds between accounts"
)

# Use with Agent
agent = Agent(
    role="Financial Advisor",
    goal="Manage finances safely",
    tools=[transfer_tool]
)
```

### AutoGen Integration

```python
from autogen import AssistantAgent, UserProxyAgent, config_list_from_json
from guardrails import guard_tool, ToolGuard
from guardrails.validators import JSONSchema

tool_guard = ToolGuard(name="autogen_tools")

@guard_tool(
    input_validators={"query": [QueryLengthValidator()]},
    output_validators=[JSONSchema(schema={"type": "object"})],
    tool_guard=tool_guard
)
def search_knowledge_base(query: str) -> dict:
    """Search knowledge base with validated query and response."""
    # Implementation
    return {"results": [], "count": 0}

# Register with AutoGen
assistant = AssistantAgent("assistant")
assistant.register_function(search_knowledge_base)
```

## OnFailAction Strategies

### EXCEPTION (Default)

Raises `ToolValidationError` when validation fails. Use for strict security requirements.

```python
@guard_tool(
    input_validators={"admin_action": [AdminAuthValidator()]},
    on_input_fail=OnFailAction.EXCEPTION
)
def delete_database(admin_action: bool) -> str:
    """Delete database - must be authorized."""
    return "Database deleted"

# Calling with unauthorized input raises ToolValidationError
try:
    delete_database(False)
except ToolValidationError as e:
    print(f"Unauthorized: {e.failures}")
```

### FIX

Automatically fixes invalid inputs using validator's fix_value. Use when validators can safely correct inputs.

```python
from guardrails.validators import ValidEmail

@guard_tool(
    input_validators={"email": [ValidEmail()]},
    on_input_fail=OnFailAction.FIX
)
def subscribe(email: str) -> dict:
    """Subscribe email - invalid emails are fixed."""
    return {"status": "subscribed", "email": email}

# "user@domain" -> "user@domain.com" (fixed by validator)
result = subscribe("user@domain")
```

### FILTER

Removes invalid data. Use for optional fields or lists.

```python
@guard_tool(
    input_validators={"tags": [ValidTags()]},
    on_input_fail=OnFailAction.FILTER
)
def add_tags(tags: list) -> dict:
    """Add tags - invalid tags are filtered out."""
    return {"tags_added": tags}

# ["valid", "invalid!", "good"] -> ["valid", "good"]
result = add_tags(["valid", "invalid!", "good"])
```

### REFRAIN

Returns None/empty value without executing tool. Use when execution is unsafe.

```python
@guard_tool(
    input_validators={"file_path": [SafeFilePathValidator()]},
    on_input_fail=OnFailAction.REFRAIN
)
def read_file(file_path: str) -> str:
    """Read file - unsafe paths are refrained."""
    with open(file_path) as f:
        return f.read()

# Unsafe path returns None without execution
result = read_file("../../etc/passwd")  # Returns None
```

### NOOP

Allows execution despite validation failure. Use for logging/monitoring only.

```python
@guard_tool(
    input_validators={"experimental": [ExperimentalValidator()]},
    on_input_fail=OnFailAction.NOOP
)
def experimental_feature(experimental: bool) -> dict:
    """Experimental feature - warnings logged but execution proceeds."""
    return {"result": "experimental"}
```

## Execution Tracking and Monitoring

### ToolGuard Manager

```python
from guardrails import ToolGuard, guard_tool

tool_guard = ToolGuard(name="api_tools")

@guard_tool(
    input_validators={"user_id": [UserValidator()]},
    tool_guard=tool_guard
)
def get_user_profile(user_id: int) -> dict:
    return {"id": user_id, "name": "User"}

@guard_tool(
    output_validators=[APIResponseValidator()],
    tool_guard=tool_guard
)
def list_users() -> list:
    return [{"id": 1, "name": "User 1"}]

# Get execution history
all_history = tool_guard.get_execution_history()
get_user_history = tool_guard.get_execution_history("get_user_profile")

# Get statistics
stats = tool_guard.get_statistics()
print(f"Total executions: {stats['total_executions']}")
print(f"Pass rate: {stats['validation_pass_rate']}")
print(f"Input pass rate: {stats['input_pass_rate']}")
print(f"Output pass rate: {stats['output_pass_rate']}")

# Per-tool statistics
user_stats = tool_guard.get_statistics("get_user_profile")
```

### Validation Result Details

```python
# Inspect individual execution results
for result in tool_guard.execution_history:
    print(f"Tool: {result.tool_name}")
    print(f"Valid: {result.validation_passed}")
    print(f"Input failures: {result.input_failures}")
    print(f"Output failures: {result.output_failures}")
    print(f"Execution time: {result.execution_time_ms}ms")
    print(f"Input action: {result.input_on_fail_action}")
    print(f"Output action: {result.output_on_fail_action}")
    
    # Serialize for logging/monitoring
    result_dict = result.to_dict()
```

## Advanced Patterns

### Parameter-Level Validation

Validate individual parameters differently:

```python
from guardrails import guard_tool
from guardrails.validators import ValidRange, EmailValidator

@guard_tool(
    input_validators={
        "age": [ValidRange(min=0, max=150)],
        "email": [EmailValidator()],
        "score": [ValidRange(min=0, max=100)]
    },
    on_input_fail=OnFailAction.FIX
)
def create_user(age: int, email: str, score: float) -> dict:
    """Create user with per-parameter validation."""
    return {"status": "created"}
```

### Multiple Validators Per Parameter

Chain validators for comprehensive validation:

```python
from guardrails.validators import ValidLength, ValidFormat

@guard_tool(
    input_validators={
        "password": [
            ValidLength(min=8, max=100),
            ValidFormat(pattern=r"[A-Z].*[0-9]")  # At least one uppercase and number
        ]
    },
    on_input_fail=OnFailAction.EXCEPTION
)
def set_password(password: str) -> dict:
    """Set password with multi-step validation."""
    return {"status": "password_set"}
```

### Async Tool Support

Full support for async tools in agent workflows:

```python
import asyncio
from guardrails import guard_tool

@guard_tool(
    input_validators={"api_key": [APIKeyValidator()]},
    output_validators=[JSONValidator()],
    on_input_fail=OnFailAction.EXCEPTION
)
async def fetch_remote_data(api_key: str) -> dict:
    """Fetch data from remote API asynchronously."""
    # Async implementation
    import aiohttp
    async with aiohttp.ClientSession() as session:
        # API call
        return {"data": []}

# Use in agent
async def agent_workflow():
    result = await fetch_remote_data("valid-key")
    return result
```

## Error Handling

### Catching Validation Errors

```python
from guardrails import guard_tool, ToolValidationError

@guard_tool(
    input_validators={"amount": [ValidRange(min=0, max=1000)]},
    on_input_fail=OnFailAction.EXCEPTION
)
def process_payment(amount: float) -> dict:
    return {"status": "processed"}

# Handle validation errors in agent logic
try:
    result = process_payment(2000)  # Exceeds max
except ToolValidationError as e:
    print(f"Validation failed: {e.failures}")
    # Fall back or retry with agent
```

### Metadata for Custom Logic

```python
@guard_tool(
    input_validators={"data": [CustomValidator()]},
    metadata={"severity": "high", "context": "payment"}
)
def payment_tool(data: dict) -> dict:
    """Tool with metadata for validator access."""
    return {"status": "complete"}
```

## Observability

### OpenTelemetry Integration

Tool execution is automatically traced with OpenTelemetry/OpenInference:

```python
from guardrails import guard_tool

@guard_tool(
    input_validators={"query": [QueryValidator()]},
    output_validators=[ResponseValidator()]
)
def search(query: str) -> dict:
    """Search - automatically traced."""
    return {"results": []}

# Spans created:
# - span name: "tool/search"
# - attributes:
#   - type: "guardrails/tool"
#   - tool.name: "search"
#   - tool.id: "<unique-id>"
#   - tool.input_validation_passed: true/false
#   - tool.output_validation_passed: true/false
#   - tool.input_failures: {...}
#   - tool.output_failures: {...}
```

### Disabling Tracing

```python
import os

# Disable tracing globally
os.environ["GUARDRAILS_DISABLE_TRACING"] = "true"

# Or via settings
from guardrails.settings import settings
settings.disable_tracing = True
```

## Security Best Practices

1. **Use EXCEPTION for security-critical operations**: Admin functions, financial transactions
2. **Use FIX for user convenience**: Email normalization, whitespace trimming
3. **Validate both inputs and outputs**: Prevent injection from external APIs
4. **Monitor execution statistics**: Track validation failures for anomaly detection
5. **Use ToolGuard manager**: Centralized tracking across all agent tools
6. **Implement per-parameter validation**: Catch malformed arguments early
7. **Chain validators**: Use multiple validators for defense-in-depth
8. **Enable tracing**: Integrate with observability systems for audit trails

## Testing Tools

```python
import pytest
from guardrails import guard_tool, ToolValidationError

@guard_tool(
    input_validators={"x": [RangeValidator(0, 100)]},
    on_input_fail=OnFailAction.EXCEPTION
)
def process(x: int) -> int:
    return x * 2

def test_valid_input():
    assert process(50) == 100

def test_invalid_input():
    with pytest.raises(ToolValidationError):
        process(150)

def test_with_tool_guard():
    from guardrails import ToolGuard
    
    guard = ToolGuard()
    
    @guard_tool(
        input_validators={"x": [RangeValidator(0, 100)]},
        tool_guard=guard
    )
    def tracked_process(x: int) -> int:
        return x * 2
    
    tracked_process(50)
    assert len(guard.execution_history) == 1
    assert guard.execution_history[0].input_validation_passed
```

## Summary

The `@guard_tool` decorator and `ToolGuard` manager provide comprehensive input/output validation for agent tools, addressing critical security concerns while maintaining flexibility in failure handling. Integrate them into your agent frameworks for more secure and observable autonomous workflows.
