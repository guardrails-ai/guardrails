# @guard_tool Architecture and Design

## Overview

The `@guard_tool` decorator and `ToolGuard` manager provide a comprehensive framework for validating tool inputs and outputs in autonomous agent systems. This document describes the architecture, design decisions, and implementation details.

## Components

### 1. ToolValidationError

Custom exception raised when validation fails with `OnFailAction.EXCEPTION`.

```python
class ToolValidationError(ValidationError):
    def __init__(self, message: str, failures: Dict[str, Any]):
        self.failures = failures  # Dict mapping parameter names to error messages
```

**Usage**: Catches validation failures in tool execution.

### 2. ToolValidationResult

Immutable result object capturing a single tool execution's validation outcome.

```python
class ToolValidationResult:
    tool_id: str                          # Unique execution identifier
    tool_name: str                        # Name of the tool function
    input_validation_passed: bool         # Input validation success
    output_validation_passed: bool        # Output validation success
    input_failures: Dict[str, Any]        # Input validation errors by parameter
    output_failures: Dict[str, Any]       # Output validation errors
    validated_inputs: Dict[str, Any]      # Processed input parameters
    validated_output: Any                 # Processed output value
    execution_time_ms: float              # Tool execution duration
    input_on_fail_action: OnFailAction    # Input failure strategy used
    output_on_fail_action: OnFailAction   # Output failure strategy used
    
    @property
    def validation_passed() -> bool       # True if all validations passed
    
    def to_dict() -> Dict[str, Any]       # Serialize to dictionary
```

**Purpose**: Provides detailed execution telemetry for monitoring and debugging.

### 3. ToolGuard Manager

Centralized manager for tool registration, execution tracking, and statistics.

```python
class ToolGuard:
    name: str                                    # Manager instance name
    execution_history: List[ToolValidationResult]  # All executions
    _tool_validators: Dict[str, Dict]            # Registered tools
    
    def register_tool(
        tool_name: str,
        input_validators: Optional[...],
        output_validators: Optional[...],
        metadata: Optional[Dict]
    ) -> str                                     # Returns tool_id
    
    def get_execution_history(
        tool_name: Optional[str]
    ) -> List[ToolValidationResult]              # Optionally filtered
    
    def get_statistics(
        tool_name: Optional[str]
    ) -> Dict[str, float]                        # Validation rates
```

**Responsibilities**:
- Registers tools and maintains validator mappings
- Records execution history
- Calculates validation statistics
- Provides filtering and querying capabilities

### 4. @guard_tool Decorator

Function decorator that wraps tool execution with validation logic.

```python
@guard_tool(
    input_validators: Optional[Union[List[Validator], Dict[str, List[Validator]]]],
    output_validators: Optional[List[Validator]],
    metadata: Optional[Dict[str, Any]],
    tool_guard: Optional[ToolGuard],
    on_input_fail: Optional[Union[str, OnFailAction]],
    on_output_fail: Optional[Union[str, OnFailAction]]
)
def my_tool(...) -> Any:
    ...
```

**Responsibilities**:
- Validates inputs before execution
- Executes the tool function
- Validates outputs after execution
- Handles failures according to OnFailAction
- Records execution with ToolGuard if provided
- Integrates with OpenTelemetry for tracing

## Validation Flow

### Input Validation Phase

```
1. Bind function arguments from *args, **kwargs
2. Apply defaults to complete parameter set
3. For each input_validator:
   a. Call validator.validate(value, metadata)
   b. If FailResult:
      - Apply validator.on_fail_descriptor strategy (fix, custom, etc.)
      - Accumulate failures if strategy doesn't fix
   c. If PassResult: continue
4. If failures and on_input_fail == EXCEPTION: raise ToolValidationError
5. Otherwise: apply on_input_fail action (FIX, FILTER, REFRAIN, NOOP)
6. Return validated parameters
```

### Execution Phase

```
1. Call original function with validated inputs
2. Catch any exceptions and propagate
3. Return raw output for output validation
```

### Output Validation Phase

```
1. Apply same logic as input validation
2. Return validated output or apply on_output_fail action
```

### Execution Recording

```
If tool_guard provided:
1. Create ToolValidationResult with all metrics
2. Record in tool_guard.execution_history
3. Execution is tracked for statistics and monitoring
```

## OnFailAction Strategies

### EXCEPTION

- **Behavior**: Raises `ToolValidationError` immediately
- **Use Case**: Security-critical operations, admin functions
- **Impact**: Stops execution, error propagates to agent

### FIX

- **Behavior**: Uses validator's `fix_value` to correct input
- **Use Case**: Normalization (email lowercase, whitespace trim)
- **Impact**: Execution proceeds with fixed value
- **Requirement**: Validator must provide `fix_value`

### FILTER

- **Behavior**: Removes invalid elements from collections
- **Use Case**: Filtering spam, removing invalid tags
- **Impact**: Reduces input size, execution proceeds with subset
- **Limitation**: Only meaningful for dict/list inputs

### REFRAIN

- **Behavior**: Returns None/empty without executing function
- **Use Case**: Unsafe operations (file access, dangerous commands)
- **Impact**: Function not called, None returned to agent
- **Safety**: Prevents execution on suspicious input

### NOOP

- **Behavior**: Logs warning but executes function with original value
- **Use Case**: Monitoring, logging, experimental features
- **Impact**: Execution proceeds, validation failure is recorded
- **Use Carefully**: Validation has no effect

### REASK

- **Behavior**: For tools, treated as EXCEPTION (LLM interaction not available)
- **Note**: Designed for Guard context, not applicable to tools

## Validation Patterns

### Parameter-Level Validation

Different validators per parameter:

```python
@guard_tool(
    input_validators={
        "email": [EmailValidator()],
        "age": [ValidRange(0, 150)],
        "country": [CountryValidator()]
    }
)
```

**Flow**: Each parameter validated independently, failures aggregated.

### Aggregate Validation

Single validator for entire arguments dict:

```python
@guard_tool(
    input_validators=[
        CrossParameterValidator(),  # Validates relationships
        ContextValidator()          # Validates in context
    ]
)
```

**Flow**: Validators receive entire parameters dict, can validate interdependencies.

### Chained Validators

Multiple validators on single parameter:

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
```

**Flow**: Validators applied sequentially, failures accumulated.

## OpenTelemetry Integration

### Span Creation

```
Span Name: "tool/{tool_name}"
Span Type: guardrails/tool
Attributes:
  - type: "guardrails/tool"
  - tool.name: {tool_name}
  - tool.id: {unique_id}
  - tool.input_validation_passed: bool
  - tool.output_validation_passed: bool
  - tool.input_failures: {failures_dict}
  - tool.output_failures: {failures_dict}
  - input.mime_type: "application/json"
  - input.value: {serialized_inputs}
  - output.mime_type: "application/json"
  - output.value: {serialized_output}
  - openinference_span_kind: "TOOL"
```

### Status Codes

```python
if validation_passed:
    span.set_status(StatusCode.OK)
else:
    span.set_status(StatusCode.ERROR, description="Tool validation failed")
```

### Context Propagation

OpenTelemetry context is automatically preserved through tool execution for integration with upstream agent frameworks.

## Thread Safety and Async

### Synchronous Execution

- Function validated and executed in calling thread
- No special synchronization required
- ToolGuard.execution_history is list-safe (thread-safe append)

### Asynchronous Execution

- Async function detected via `inspect.iscoroutinefunction()`
- Validation runs in caller's event loop
- Function awaited normally
- All tracing and recording works with async/await

```python
@guard_tool(...)
async def async_tool(...):
    await asyncio.sleep(1)
    return result
```

## Performance Considerations

### Validation Overhead

1. **Signature binding**: O(n) where n = number of parameters
2. **Validator execution**: Depends on validator complexity
3. **Failure accumulation**: O(n * m) where m = number of validators
4. **Tracing**: Span creation adds ~1-5ms overhead

### Optimization Strategies

1. **Disable tracing**: Set `GUARDRAILS_DISABLE_TRACING=true` in production
2. **Minimal validators**: Only validate critical parameters
3. **Early exit**: Use EXCEPTION for fail-fast on security checks
4. **Cached validators**: Reuse validator instances across multiple tools

## Error Handling

### Validation Errors

```python
try:
    result = guarded_tool(...)
except ToolValidationError as e:
    print(e.failures)  # Dict[str, List[str]] - errors by parameter
```

### Tool Execution Errors

Tool exceptions during execution are not caught by @guard_tool. If tool raises exception after input validation passes:
- Exception propagates to caller
- Output validation is skipped
- Execution is NOT recorded in ToolGuard

### Telemetry Errors

If tracing fails, execution continues (non-fatal). Errors logged but don't block tool execution.

## Metadata Flow

```python
@guard_tool(
    input_validators=[...],
    metadata={
        "context": "payment",
        "sensitivity": "high",
        "source": "external_api"
    }
)
```

Metadata passed to all validators via `validator.validate(value, metadata)`.

## Extension Points

### Custom Validators

Validators inheriting from `guardrails.Validator` work seamlessly:

```python
class CustomValidator(Validator):
    def _validate(self, value, metadata):
        # Custom logic
        return PassResult() or FailResult(error_message="...", fix_value=...)
```

### Custom OnFailActions

Extend `OnFailAction` enum for domain-specific behaviors (in future versions).

### Custom Telemetry

Replace default tracing by setting `settings.disable_tracing = True` and implementing custom span creation.

## Comparison with Guard

| Feature | @guard_tool | Guard |
|---------|-----------|-------|
| Purpose | Tool validation | LLM output validation |
| Input | Function arguments | LLM API response |
| Tracing | OpenInference TOOL spans | OpenInference GUARDRAIL spans |
| Reask | Not applicable | Supported (LLM regen) |
| Execution Tracking | ToolGuard manager | Guard.history |
| Failure Handling | OnFailAction | OnFailAction |
| Async | Native async support | AsyncGuard variant |

## Security Properties

### Input Injection Prevention

- Validates LLM-generated arguments before tool execution
- Prevents parameter injection, path traversal, etc.
- Configurable per-parameter validation strategies

### Output Injection Prevention

- Validates tool outputs before returning to LLM
- Prevents prompt injection from poisoned sources
- Catches unexpected output formats early

### Audit Trail

- Complete execution history with ToolGuard
- Tracing integration for security monitoring
- Failure tracking for anomaly detection

## Future Enhancements

1. **Custom OnFailActions**: Plugin architecture for domain-specific actions
2. **Distributed Tracing**: Enhanced propagation across service boundaries
3. **ML-based Anomaly Detection**: Auto-detect suspicious patterns
4. **Dynamic Validator Loading**: Load validators based on tool signatures
5. **Performance Profiling**: Built-in metrics for validator performance
6. **Validator Composition**: Higher-order validators for complex rules
