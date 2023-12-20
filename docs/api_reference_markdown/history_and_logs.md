# History and Logs

## Iteration

#### logs()

```python
@property
def logs() -> Stack[str]
```

Returns the logs from this iteration as a stack.

#### tokens\_consumed()

```python
@property
def tokens_consumed() -> Optional[int]
```

Returns the total number of tokens consumed during this
iteration.

#### prompt\_tokens\_consumed()

```python
@property
def prompt_tokens_consumed() -> Optional[int]
```

Returns the number of prompt/input tokens consumed during this
iteration.

#### completion\_tokens\_consumed()

```python
@property
def completion_tokens_consumed() -> Optional[int]
```

Returns the number of completion/output tokens consumed during this
iteration.

#### raw\_output()

```python
@property
def raw_output() -> Optional[str]
```

The exact output from the LLM.

#### parsed\_output()

```python
@property
def parsed_output() -> Optional[Union[str, Dict]]
```

The output from the LLM after undergoing parsing but before
validation.

#### validation\_output()

```python
@property
def validation_output() -> Optional[Union[ReAsk, str, Dict]]
```

The output from the validation process.

Could be a combination of valid output and ReAsks

#### validated\_output()

```python
@property
def validated_output() -> Optional[Union[str, Dict]]
```

The valid output from the LLM after undergoing validation.

Could be only a partial structure if field level reasks occur.
Could contain fixed values.

#### reasks()

```python
@property
def reasks() -> Sequence[ReAsk]
```

Reasks generated during validation.

These would be incorporated into the prompt or the next LLM
call.

#### validator\_logs()

```python
@property
def validator_logs() -> List[ValidatorLogs]
```

The results of each individual validation performed on the LLM
response during this iteration.

#### error()

```python
@property
def error() -> Optional[str]
```

The error message from any exception that raised and interrupted
this iteration.

#### exception()

```python
@property
def exception() -> Optional[Exception]
```

The exception that interrupted this iteration.

#### failed\_validations()

```python
@property
def failed_validations() -> List[ValidatorLogs]
```

The validator logs for any validations that failed during this
iteration.

#### status()

```python
@property
def status() -> str
```

Representation of the end state of this iteration.

OneOf: pass, fail, error, not run

## Call

#### prompt()

```python
@property
def prompt() -> Optional[str]
```

The prompt as provided by the user when intializing or calling the
Guard.

#### prompt\_params()

```python
@property
def prompt_params() -> Optional[Dict]
```

The prompt parameters as provided by the user when intializing or
calling the Guard.

#### compiled\_prompt()

```python
@property
def compiled_prompt() -> Optional[str]
```

The initial compiled prompt that was passed to the LLM on the first
call.

#### reask\_prompts()

```python
@property
def reask_prompts() -> Stack[Optional[str]]
```

The compiled prompts used during reasks.

Does not include the initial prompt.

#### instructions()

```python
@property
def instructions() -> Optional[str]
```

The instructions as provided by the user when intializing or calling
the Guard.

#### compiled\_instructions()

```python
@property
def compiled_instructions() -> Optional[str]
```

The initial compiled instructions that were passed to the LLM on the
first call.

#### reask\_instructions()

```python
@property
def reask_instructions() -> Stack[str]
```

The compiled instructions used during reasks.

Does not include the initial instructions.

#### logs()

```python
@property
def logs() -> Stack[str]
```

Returns all logs from all iterations as a stack.

#### tokens\_consumed()

```python
@property
def tokens_consumed() -> Optional[int]
```

Returns the total number of tokens consumed during all iterations
with this call.

#### prompt\_tokens\_consumed()

```python
@property
def prompt_tokens_consumed() -> Optional[int]
```

Returns the total number of prompt tokens consumed during all
iterations with this call.

#### completion\_tokens\_consumed()

```python
@property
def completion_tokens_consumed() -> Optional[int]
```

Returns the total number of completion tokens consumed during all
iterations with this call.

#### raw\_outputs()

```python
@property
def raw_outputs() -> Stack[str]
```

The exact outputs from all LLM calls.

#### parsed\_outputs()

```python
@property
def parsed_outputs() -> Stack[Union[str, Dict]]
```

The outputs from the LLM after undergoing parsing but before
validation.

#### validation\_output()

```python
@property
def validation_output() -> Optional[Union[str, Dict, ReAsk]]
```

The cumulative validation output across all current iterations.

Could contain ReAsks.

#### fixed\_output()

```python
@property
def fixed_output() -> Optional[Union[str, Dict]]
```

The cumulative validation output across all current iterations with
any automatic fixes applied.

#### validated\_output()

```python
@property
def validated_output() -> Optional[Union[str, Dict]]
```

The output from the LLM after undergoing validation.

This will only have a value if the Guard is in a passing state.

#### reasks()

```python
@property
def reasks() -> Stack[ReAsk]
```

Reasks generated during validation that could not be automatically
fixed.

These would be incorporated into the prompt for the next LLM
call if additional reasks were granted.

#### validator\_logs()

```python
@property
def validator_logs() -> Stack[ValidatorLogs]
```

The results of each individual validation performed on the LLM
responses during all iterations.

#### error()

```python
@property
def error() -> Optional[str]
```

The error message from any exception that raised and interrupted the
run.

#### exception()

```python
@property
def exception() -> Optional[Exception]
```

The exception that interrupted the run.

#### failed\_validations()

```python
@property
def failed_validations() -> Stack[ValidatorLogs]
```

The validator logs for any validations that failed during the
entirety of the run.

#### status()

```python
@property
def status() -> str
```

Returns the cumulative status of the run based on the validity of
the final merged output.

#### tree()

```python
@property
def tree() -> Tree
```

Returns the tree.

## Outputs

#### failed\_validations()

```python
@property
def failed_validations() -> List[ValidatorLogs]
```

Returns the validator logs for any validation that failed.

#### status()

```python
@property
def status() -> str
```

Representation of the end state of the validation run.

OneOf: pass, fail, error, not run

