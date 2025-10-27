# History and Logs

## Call

```python
class Call(ICall, ArbitraryModel)
```

A Call represents a single execution of a Guard. One Call is created
each time the user invokes the `Guard.__call__`, `Guard.parse`, or
`Guard.validate` method.

**Attributes**:

- `iterations` _Stack[Iteration]_ - A stack of iterations
  for the initial validation round
  and one for each reask that occurs during a Call.
- `inputs` _CallInputs_ - The inputs as passed in to
  `Guard.__call__`, `Guard.parse`, or `Guard.validate`
- `exception` _Optional[Exception]_ - The exception that interrupted
  the Guard execution.

#### prompt\_params

```python
@property
def prompt_params() -> Optional[Dict]
```

The prompt parameters as provided by the user when initializing or
calling the Guard.

#### messages

```python
@property
def messages() -> Optional[Union[Messages, list[dict[str, str]]]]
```

The messages as provided by the user when initializing or calling
the Guard.

#### compiled\_messages

```python
@property
def compiled_messages() -> Optional[list[dict[str, str]]]
```

The initial compiled messages that were passed to the LLM on the
first call.

#### reask\_messages

```python
@property
def reask_messages() -> Stack[Messages]
```

The compiled messages used during reasks.

Does not include the initial messages.

#### logs

```python
@property
def logs() -> Stack[str]
```

Returns all logs from all iterations as a stack.

#### tokens\_consumed

```python
@property
def tokens_consumed() -> Optional[int]
```

Returns the total number of tokens consumed during all iterations
with this call.

#### prompt\_tokens\_consumed

```python
@property
def prompt_tokens_consumed() -> Optional[int]
```

Returns the total number of prompt tokens consumed during all
iterations with this call.

#### completion\_tokens\_consumed

```python
@property
def completion_tokens_consumed() -> Optional[int]
```

Returns the total number of completion tokens consumed during all
iterations with this call.

#### raw\_outputs

```python
@property
def raw_outputs() -> Stack[str]
```

The exact outputs from all LLM calls.

#### parsed\_outputs

```python
@property
def parsed_outputs() -> Stack[Union[str, List, Dict]]
```

The outputs from the LLM after undergoing parsing but before
validation.

#### validation\_response

```python
@property
def validation_response() -> Optional[Union[str, List, Dict, ReAsk]]
```

The aggregated responses from the validation process across all
iterations within the current call.

This value could contain ReAsks.

#### fixed\_output

```python
@property
def fixed_output() -> Optional[Union[str, List, Dict]]
```

The cumulative output from the validation process across all current
iterations with any automatic fixes applied.

Could still contain ReAsks if a fix was not available.

#### guarded\_output

```python
@property
def guarded_output() -> Optional[Union[str, List, Dict]]
```

The complete validated output after all stages of validation are
completed.

This property contains the aggregate validated output after all
validation stages have been completed. Some values in the
validated output may be "fixed" values that were corrected
during validation.

This will only have a value if the Guard is in a passing state
OR if the action is no-op.

#### reasks

```python
@property
def reasks() -> Stack[ReAsk]
```

Reasks generated during validation that could not be automatically
fixed.

These would be incorporated into the prompt for the next LLM
call if additional reasks were granted.

#### validator\_logs

```python
@property
def validator_logs() -> Stack[ValidatorLogs]
```

The results of each individual validation performed on the LLM
responses during all iterations.

#### error

```python
@property
def error() -> Optional[str]
```

The error message from any exception that raised and interrupted the
run.

#### failed\_validations

```python
@property
def failed_validations() -> Stack[ValidatorLogs]
```

The validator logs for any validations that failed during the
entirety of the run.

#### status

```python
@property
def status() -> str
```

Returns the cumulative status of the run based on the validity of
the final merged output.

#### tree

```python
@property
def tree() -> Tree
```

Returns the tree.

## Iteration

```python
class Iteration(IIteration, ArbitraryModel)
```

An Iteration represents a single iteration of the validation loop
including a single call to the LLM if applicable.

**Attributes**:

- `id` _str_ - The unique identifier for the iteration.
- `call_id` _str_ - The unique identifier for the Call
  that this iteration is a part of.
- `index` _int_ - The index of this iteration within the Call.
- `inputs` _Inputs_ - The inputs for the validation loop.
- `outputs` _Outputs_ - The outputs from the validation loop.

#### logs

```python
@property
def logs() -> Stack[str]
```

Returns the logs from this iteration as a stack.

#### tokens\_consumed

```python
@property
def tokens_consumed() -> Optional[int]
```

Returns the total number of tokens consumed during this
iteration.

#### prompt\_tokens\_consumed

```python
@property
def prompt_tokens_consumed() -> Optional[int]
```

Returns the number of prompt/input tokens consumed during this
iteration.

#### completion\_tokens\_consumed

```python
@property
def completion_tokens_consumed() -> Optional[int]
```

Returns the number of completion/output tokens consumed during this
iteration.

#### raw\_output

```python
@property
def raw_output() -> Optional[str]
```

The exact output from the LLM.

#### parsed\_output

```python
@property
def parsed_output() -> Optional[Union[str, List, Dict]]
```

The output from the LLM after undergoing parsing but before
validation.

#### validation\_response

```python
@property
def validation_response() -> Optional[Union[ReAsk, str, List, Dict]]
```

The response from a single stage of validation.

Validation response is the output of a single stage of validation
and could be a combination of valid output and reasks.
Note that a Guard may run validation multiple times if reasks occur.
To access the final output after all steps of validation are completed,
check out `Call.guarded_output`."

#### guarded\_output

```python
@property
def guarded_output() -> Optional[Union[str, List, Dict]]
```

Any valid values after undergoing validation.

Some values in the validated output may be "fixed" values that
were corrected during validation. This property may be a partial
structure if field level reasks occur.

#### reasks

```python
@property
def reasks() -> Sequence[ReAsk]
```

Reasks generated during validation.

These would be incorporated into the prompt or the next LLM
call.

#### validator\_logs

```python
@property
def validator_logs() -> List[ValidatorLogs]
```

The results of each individual validation performed on the LLM
response during this iteration.

#### error

```python
@property
def error() -> Optional[str]
```

The error message from any exception that raised and interrupted
this iteration.

#### exception

```python
@property
def exception() -> Optional[Exception]
```

The exception that interrupted this iteration.

#### failed\_validations

```python
@property
def failed_validations() -> List[ValidatorLogs]
```

The validator logs for any validations that failed during this
iteration.

#### error\_spans\_in\_output

```python
@property
def error_spans_in_output() -> List[ErrorSpan]
```

The error spans from the LLM response.

These indices are relative to the complete LLM output.

#### status

```python
@property
def status() -> str
```

Representation of the end state of this iteration.

OneOf: pass, fail, error, not run

## Inputs

```python
class Inputs(IInputs, ArbitraryModel)
```

Inputs represent the input data that is passed into the validation loop.

**Attributes**:

- `llm_api` _Optional[PromptCallableBase]_ - The constructed class
  for calling the LLM.
- `llm_output` _Optional[str]_ - The string output from an
  external LLM call provided by the user via Guard.parse.
- `messages` _Optional[List[Dict]]_ - The message history
  provided by the user for chat model calls.
- `prompt_params` _Optional[Dict]_ - The parameters provided
  by the user that will be formatted into the final LLM prompt.
- `num_reasks` _Optional[int]_ - The total number of reasks allowed;
  user provided or defaulted.
- `metadata` _Optional[Dict[str, Any]]_ - The metadata provided
  by the user to be used during validation.
- `full_schema_reask` _Optional[bool]_ - Whether reasks we
  performed across the entire schema or at the field level.
- `stream` _Optional[bool]_ - Whether or not streaming was used.

## Outputs

```python
class Outputs(IOutputs, ArbitraryModel)
```

Outputs represent the data that is output from the validation loop.

**Attributes**:

- `llm_response_info` _Optional[LLMResponse]_ - Information from the LLM response
- `raw_output` _Optional[str]_ - The exact output from the LLM.
- `parsed_output` _Optional[Union[str, List, Dict]]_ - The output parsed from the LLM
  response as it was passed into validation.
- `validation_response` _Optional[Union[str, ReAsk, List, Dict]]_ - The response
  from the validation process.
- `guarded_output` _Optional[Union[str, List, Dict]]_ - Any valid values after
  undergoing validation.
  Some values may be "fixed" values that were corrected during validation.
  This property may be a partial structure if field level reasks occur.
- `reasks` _List[ReAsk]_ - Information from the validation process used to construct
  a ReAsk to the LLM on validation failure. Default [].
- `validator_logs` _List[ValidatorLogs]_ - The results of each individual
  validation. Default [].
- `error` _Optional[str]_ - The error message from any exception that raised
  and interrupted the process.
- `exception` _Optional[Exception]_ - The exception that interrupted the process.

#### failed\_validations

```python
@property
def failed_validations() -> List[ValidatorLogs]
```

Returns the validator logs for any validation that failed.

#### error\_spans\_in\_output

```python
@property
def error_spans_in_output() -> List[ErrorSpan]
```

The error spans from the LLM response.

These indices are relative to the complete LLM output.

#### status

```python
@property
def status() -> str
```

Representation of the end state of the validation run.

OneOf: pass, fail, error, not run

## CallInputs

```python
class CallInputs(Inputs, ICallInputs, ArbitraryModel)
```

CallInputs represent the input data that is passed into the Guard from
the user. Inherits from Inputs with the below overrides and additional
attributes.

**Attributes**:

- `llm_api` _Optional[Callable[[Any], Awaitable[Any]]]_ - The LLM function
  provided by the user during Guard.__call__ or Guard.parse.
- `messages` _Optional[dict[str, str]]_ - The messages as provided by the user.
- `args` _List[Any]_ - Additional arguments for the LLM as provided by the user.
  Default [].
- `kwargs` _Dict[str, Any]_ - Additional keyword-arguments for
  the LLM as provided by the user. Default {}.

