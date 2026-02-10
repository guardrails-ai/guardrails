# Guards

## Guard

```python
class Guard(IGuard, Generic[OT])
```

The Guard class.

This class is the main entry point for using Guardrails. It can be
initialized by one of the following patterns:

- `Guard().use(...)`
- `Guard.for_string(...)`
- `Guard.for_pydantic(...)`
- `Guard.for_rail(...)`
- `Guard.for_rail_string(...)`

The `__call__`
method functions as a wrapper around LLM APIs. It takes in an LLM
API, and optional prompt parameters, and returns a ValidationOutcome
class that contains the raw output from
the LLM, the validated output, as well as other helpful information.

#### \_\_init\_\_

```python
def __init__(*,
             id: Optional[str] = None,
             name: Optional[str] = None,
             description: Optional[str] = None,
             validators: Optional[List[ValidatorReference]] = None,
             output_schema: Optional[Dict[str, Any]] = None,
             base_url: Optional[str] = None,
             api_key: Optional[str] = None,
             history_max_length: Optional[int] = None,
             use_server: Optional[bool] = None)
```

Initialize the Guard with serialized validator references and an
output schema.

Output schema must be a valid JSON Schema.

#### configure

```python
def configure(*,
              num_reasks: Optional[int] = None,
              tracer: Optional[Tracer] = None,
              allow_metrics_collection: Optional[bool] = None)
```

Configure the Guard.

**Arguments**:

- `num_reasks` _int, optional_ - The max times to re-ask the LLM
  if validation fails. Defaults to None.
- `tracer` _Tracer, optional_ - An OpenTelemetry tracer to use for
  sending traces to your OpenTelemetry sink. Defaults to None.
- `allow_metrics_collection` _bool, optional_ - Whether to allow
  Guardrails to collect anonymous metrics.
  Defaults to None, and falls back to waht is
  set via the `guardrails configure` command.

#### for\_rail

```python
@classmethod
def for_rail(cls,
             rail_file: str,
             *,
             num_reasks: Optional[int] = None,
             tracer: Optional[Tracer] = None,
             name: Optional[str] = None,
             description: Optional[str] = None)
```

Create a Guard using a `.rail` file to specify the output schema,
prompt, etc.

**Arguments**:

- `rail_file` - The path to the `.rail` file.
- `num_reasks` _int, optional_ - The max times to re-ask the LLM if validation fails. Deprecated
- `tracer` _Tracer, optional_ - An OpenTelemetry tracer to use for metrics and traces. Defaults to None.
- `name` _str, optional_ - A unique name for this Guard. Defaults to `gr-` + the object id.
- `description` _str, optional_ - A description for this Guard. Defaults to None.
  

**Returns**:

  An instance of the `Guard` class.

#### for\_rail\_string

```python
@classmethod
def for_rail_string(cls,
                    rail_string: str,
                    *,
                    num_reasks: Optional[int] = None,
                    tracer: Optional[Tracer] = None,
                    name: Optional[str] = None,
                    description: Optional[str] = None)
```

Create a Guard using a `.rail` string to specify the output schema,
prompt, etc..

**Arguments**:

- `rail_string` - The `.rail` string.
- `num_reasks` _int, optional_ - The max times to re-ask the LLM if validation fails. Deprecated
- `tracer` _Tracer, optional_ - An OpenTelemetry tracer to use for metrics and traces. Defaults to None.
- `name` _str, optional_ - A unique name for this Guard. Defaults to `gr-` + the object id.
- `description` _str, optional_ - A description for this Guard. Defaults to None.
  

**Returns**:

  An instance of the `Guard` class.

#### for\_pydantic

```python
@classmethod
def for_pydantic(cls,
                 output_class: ModelOrListOfModels,
                 *,
                 num_reasks: Optional[int] = None,
                 reask_messages: Optional[List[Dict]] = None,
                 messages: Optional[List[Dict]] = None,
                 tracer: Optional[Tracer] = None,
                 name: Optional[str] = None,
                 description: Optional[str] = None,
                 output_formatter: Optional[Union[str, BaseFormatter]] = None)
```

Create a Guard instance using a Pydantic model to specify the output
schema.

**Arguments**:

- `output_class` - (Union[Type[BaseModel], List[Type[BaseModel]]]): The pydantic model that describes
  the desired structure of the output.
- `messages` _List[Dict], optional_ - A list of messages to give to the llm. Defaults to None.
- `reask_messages` _List[Dict], optional_ - A list of messages to use during reasks. Defaults to None.
- `num_reasks` _int, optional_ - The max times to re-ask the LLM if validation fails. Deprecated
- `tracer` _Tracer, optional_ - An OpenTelemetry tracer to use for metrics and traces. Defaults to None.
- `name` _str, optional_ - A unique name for this Guard. Defaults to `gr-` + the object id.
- `description` _str, optional_ - A description for this Guard. Defaults to None.
- `output_formatter` _str | Formatter, optional_ - 'none' (default), 'jsonformer', or a Guardrails Formatter.

#### for\_string

```python
@classmethod
def for_string(cls,
               validators: Sequence[Validator],
               *,
               string_description: Optional[str] = None,
               reask_messages: Optional[List[Dict]] = None,
               messages: Optional[List[Dict]] = None,
               num_reasks: Optional[int] = None,
               tracer: Optional[Tracer] = None,
               name: Optional[str] = None,
               description: Optional[str] = None)
```

Create a Guard instance for a string response.

**Arguments**:

- `validators` - (List[Validator]): The list of validators to apply to the string output.
- `string_description` _str, optional_ - A description for the string to be generated. Defaults to None.
- `messages` _List[Dict], optional_ - A list of messages to pass to llm. Defaults to None.
- `reask_messages` _List[Dict], optional_ - A list of messages to use during reasks. Defaults to None.
- `num_reasks` _int, optional_ - The max times to re-ask the LLM if validation fails. Deprecated
- `tracer` _Tracer, optional_ - An OpenTelemetry tracer to use for metrics and traces. Defaults to None.
- `name` _str, optional_ - A unique name for this Guard. Defaults to `gr-` + the object id.
- `description` _str, optional_ - A description for this Guard. Defaults to None.

#### \_\_call\_\_

```python
def __call__(
        llm_api: Optional[Callable] = None,
        *args,
        prompt_params: Optional[Dict] = None,
        num_reasks: Optional[int] = 1,
        messages: Optional[List[Dict]] = None,
        metadata: Optional[Dict] = None,
        full_schema_reask: Optional[bool] = None,
        **kwargs
) -> Union[ValidationOutcome[OT], Iterator[ValidationOutcome[OT]]]
```

Call the LLM and validate the output.

**Arguments**:

- `llm_api` - The LLM API to call
  (e.g. openai.completions.create or openai.Completion.acreate)
- `prompt_params` - The parameters to pass to the prompt.format() method.
- `num_reasks` - The max times to re-ask the LLM for invalid output.
- `messages` - The message history to pass to the LLM.
- `metadata` - Metadata to pass to the validators.
- `full_schema_reask` - When reasking, whether to regenerate the full schema
  or just the incorrect values.
  Defaults to `True` if a base model is provided,
  `False` otherwise.
  

**Returns**:

  ValidationOutcome

#### parse

```python
def parse(llm_output: str,
          *args,
          metadata: Optional[Dict] = None,
          llm_api: Optional[Callable] = None,
          num_reasks: Optional[int] = None,
          prompt_params: Optional[Dict] = None,
          full_schema_reask: Optional[bool] = None,
          **kwargs) -> ValidationOutcome[OT]
```

Alternate flow to using Guard where the llm_output is known.

**Arguments**:

- `llm_output` - The output being parsed and validated.
- `metadata` - Metadata to pass to the validators.
- `llm_api` - The LLM API to call
  (e.g. openai.completions.create or openai.Completion.acreate)
- `num_reasks` - The max times to re-ask the LLM for invalid output.
- `prompt_params` - The parameters to pass to the prompt.format() method.
- `full_schema_reask` - When reasking, whether to regenerate the full schema
  or just the incorrect values.
  

**Returns**:

  ValidationOutcome

#### error\_spans\_in\_output

```python
def error_spans_in_output() -> List[ErrorSpan]
```

Get the error spans in the last output.

#### use

```python
def use(*validators: Validator, on: str = "output") -> "Guard"
```

Applies validators to the property specified in the `on` argument.
Calling `Guard.use` with the same `on` value multiple times will
overwrite previously configured validators on the specified property.

**Arguments**:

- `validators` - The validators to use.
- `on` - The property to validate. Valid options include "output", "messages",
  or a JSON path starting with "$.". Defaults to "output".

#### get\_validators

```python
def get_validators(on: str) -> List[Validator]
```

The read-only counterpart to `Guard.use`.
Retrieves the validators applied to the specified property.

**Arguments**:

- `on` - The property for which to return configured validators. Valid options include "output", "messages",
  or a JSON path starting with "$.".

#### validate

```python
def validate(llm_output: str, *args, **kwargs) -> ValidationOutcome[OT]
```

#### to\_runnable

```python
def to_runnable() -> Runnable
```

Convert a Guard to a LangChain Runnable.

#### to\_dict

```python
def to_dict() -> Dict[str, Any]
```

#### json\_function\_calling\_tool

```python
def json_function_calling_tool(
        tools: Optional[list] = None) -> List[Dict[str, Any]]
```

Appends an OpenAI tool that specifies the output structure using
JSON Schema for chat models.

#### from\_dict

```python
@classmethod
def from_dict(cls, obj: Optional[Dict[str, Any]]) -> Optional["Guard"]
```

## AsyncGuard

```python
class AsyncGuard(Guard, Generic[OT])
```

The AsyncGuard class.

This class one of the main entry point for using Guardrails. It is
initialized from one of the following class methods:

- `for_rail`
- `for_rail_string`
- `for_pydantic`
- `for_string`

The `__call__`
method functions as a wrapper around LLM APIs. It takes in an Async LLM
API, and optional prompt parameters, and returns the raw output stream from
the LLM and the validated output stream.

#### for\_pydantic

```python
@classmethod
def for_pydantic(cls,
                 output_class: ModelOrListOfModels,
                 *,
                 messages: Optional[List[Dict]] = None,
                 num_reasks: Optional[int] = None,
                 reask_messages: Optional[List[Dict]] = None,
                 tracer: Optional[Tracer] = None,
                 name: Optional[str] = None,
                 description: Optional[str] = None)
```

#### for\_string

```python
@classmethod
def for_string(cls,
               validators: Sequence[Validator],
               *,
               string_description: Optional[str] = None,
               messages: Optional[List[Dict]] = None,
               reask_messages: Optional[List[Dict]] = None,
               num_reasks: Optional[int] = None,
               tracer: Optional[Tracer] = None,
               name: Optional[str] = None,
               description: Optional[str] = None)
```

#### from\_dict

```python
@classmethod
def from_dict(cls, obj: Optional[Dict[str, Any]]) -> Optional["AsyncGuard"]
```

#### use

```python
def use(*validator: Validator, on: str = "output") -> "AsyncGuard"
```

#### \_\_call\_\_

```python
@async_trace(name="/guard_call", origin="AsyncGuard.__call__")
async def __call__(
    llm_api: Optional[Callable[..., Awaitable[Any]]] = None,
    *args,
    prompt_params: Optional[Dict] = None,
    num_reasks: Optional[int] = 1,
    messages: Optional[List[Dict]] = None,
    metadata: Optional[Dict] = None,
    full_schema_reask: Optional[bool] = None,
    **kwargs
) -> Union[
        ValidationOutcome[OT],
        Awaitable[ValidationOutcome[OT]],
        AsyncIterator[ValidationOutcome[OT]],
]
```

Call the LLM and validate the output. Pass an async LLM API to
return a coroutine.

**Arguments**:

- `llm_api` - The LLM API to call
  (e.g. openai.completions.create or openai.chat.completions.create)
- `prompt_params` - The parameters to pass to the prompt.format() method.
- `num_reasks` - The max times to re-ask the LLM for invalid output.
- `messages` - The message history to pass to the LLM.
- `metadata` - Metadata to pass to the validators.
- `full_schema_reask` - When reasking, whether to regenerate the full schema
  or just the incorrect values.
  Defaults to `True` if a base model is provided,
  `False` otherwise.
  

**Returns**:

  The raw text output from the LLM and the validated output.

#### parse

```python
@async_trace(name="/guard_call", origin="AsyncGuard.parse")
async def parse(llm_output: str,
                *args,
                metadata: Optional[Dict] = None,
                llm_api: Optional[Callable[..., Awaitable[Any]]] = None,
                num_reasks: Optional[int] = None,
                prompt_params: Optional[Dict] = None,
                full_schema_reask: Optional[bool] = None,
                **kwargs) -> Awaitable[ValidationOutcome[OT]]
```

Alternate flow to using AsyncGuard where the llm_output is known.

**Arguments**:

- `llm_output` - The output being parsed and validated.
- `metadata` - Metadata to pass to the validators.
- `llm_api` - The LLM API to call
  (e.g. openai.completions.create or openai.Completion.acreate)
- `num_reasks` - The max times to re-ask the LLM for invalid output.
- `prompt_params` - The parameters to pass to the prompt.format() method.
- `full_schema_reask` - When reasking, whether to regenerate the full schema
  or just the incorrect values.
  

**Returns**:

  The validated response. This is either a string or a dictionary,
  determined by the object schema defined in the RAILspec.

#### validate

```python
@async_trace(name="/guard_call", origin="AsyncGuard.validate")
async def validate(llm_output: str, *args,
                   **kwargs) -> Awaitable[ValidationOutcome[OT]]
```

## ValidationOutcome

```python
class ValidationOutcome(IValidationOutcome, ArbitraryModel, Generic[OT])
```

The final output from a Guard execution.

**Attributes**:

- `call_id` - The id of the Call that produced this ValidationOutcome.
- `raw_llm_output` - The raw, unchanged output from the LLM call.
- `validated_output` - The validated, and potentially fixed, output from the LLM call
  after passing through validation.
- `reask` - If validation continuously fails and all allocated reasks are used,
  this field will contain the final reask that would have been sent
  to the LLM if additional reasks were available.
- `validation_passed` - A boolean to indicate whether or not the LLM output
  passed validation. If this is False, the validated_output may be invalid.
- `error` - If the validation failed, this field will contain the error message

#### from\_guard\_history

```python
@classmethod
def from_guard_history(cls, call: Call)
```

Create a ValidationOutcome from a history Call object.

