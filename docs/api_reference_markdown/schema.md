## Schema

Schema class that holds a _schema attribute.

#### from\_xml(cls, root: ET.\_Element, reask\_prompt\_template: Optional[str] = None, reask\_instructions\_template: Optional[str] = None)

```python
@classmethod
def from_xml(cls,
             root: ET._Element,
             reask_prompt_template: Optional[str] = None,
             reask_instructions_template: Optional[str] = None) -> Self
```

Create a schema from an XML element.

#### validate(iteration: Iteration, data: Any, metadata: Dict, \*\*kwargs)

```python
def validate(iteration: Iteration, data: Any, metadata: Dict, **kwargs) -> Any
```

Validate a dictionary of data against the schema.

**Arguments**:

- `data` - The data to validate.
  

**Returns**:

  The validated data.

#### async\_validate(iteration: Iteration, data: Any, metadata: Dict)

```python
async def async_validate(iteration: Iteration, data: Any,
                         metadata: Dict) -> Any
```

Asynchronously validate a dictionary of data against the schema.

**Arguments**:

- `data` - The data to validate.
  

**Returns**:

  The validated data.

#### transpile(method: str = "default")

```python
def transpile(method: str = "default") -> str
```

Convert the XML schema to a string that is used for prompting a
large language model.

**Returns**:

  The prompt.

#### parse(output: str, \*\*kwargs)

```python
def parse(output: str, **kwargs) -> Tuple[Any, Optional[Exception]]
```

Parse the output from the large language model.

**Arguments**:

- `output` - The output from the large language model.
  

**Returns**:

  The parsed output, and the exception that was raised (if any).

#### introspect(data: Any)

```python
def introspect(
        data: Any) -> Tuple[Sequence[ReAsk], Optional[Union[str, Dict]]]
```

Inspect the data for reasks.

**Arguments**:

- `data` - The data to introspect.
  

**Returns**:

  A list of ReAsk objects.

#### get\_reask\_setup(reasks: Sequence[ReAsk], original\_response: Any, use\_full\_schema: bool, prompt\_params: Optional[Dict[str, Any]] = None)

```python
def get_reask_setup(
    reasks: Sequence[ReAsk],
    original_response: Any,
    use_full_schema: bool,
    prompt_params: Optional[Dict[str, Any]] = None
) -> Tuple["Schema", Prompt, Instructions]
```

Construct a schema for reasking, and a prompt for reasking.

**Arguments**:

- `reasks` - List of tuples, where each tuple contains the path to the
  reasked element, and the ReAsk object (which contains the error
  message describing why the reask is necessary).
- `original_response` - The value that was returned from the API, with reasks.
- `use_full_schema` - Whether to use the full schema, or only the schema
  for the reasked elements.
  

**Returns**:

  The schema for reasking, and the prompt for reasking.

#### preprocess\_prompt(prompt\_callable: PromptCallableBase, instructions: Optional[Instructions], prompt: Prompt)

```python
def preprocess_prompt(prompt_callable: PromptCallableBase,
                      instructions: Optional[Instructions], prompt: Prompt)
```

Preprocess the instructions and prompt before sending it to the
model.

**Arguments**:

- `prompt_callable` - The callable to be used to prompt the model.
- `instructions` - The instructions to preprocess.
- `prompt` - The prompt to preprocess.

## JsonSchema

#### is\_valid\_fragment(fragment: str, verified: set)

```python
def is_valid_fragment(fragment: str, verified: set) -> bool
```

Check if the fragment is a somewhat valid JSON.

#### parse\_fragment(fragment: str)

```python
def parse_fragment(fragment: str)
```

Parse the fragment into a dict.

#### validate(iteration: Iteration, data: Optional[Dict[str, Any]], metadata: Dict, \*\*kwargs)

```python
def validate(iteration: Iteration, data: Optional[Dict[str, Any]],
             metadata: Dict, **kwargs) -> Any
```

Validate a dictionary of data against the schema.

**Arguments**:

- `data` - The data to validate.
  

**Returns**:

  The validated data.

#### async\_validate(iteration: Iteration, data: Optional[Dict[str, Any]], metadata: Dict)

```python
async def async_validate(iteration: Iteration, data: Optional[Dict[str, Any]],
                         metadata: Dict) -> Any
```

Validate a dictionary of data against the schema.

**Arguments**:

- `data` - The data to validate.
  

**Returns**:

  The validated data.

## StringSchema

#### validate(iteration: Iteration, data: Any, metadata: Dict, \*\*kwargs)

```python
def validate(iteration: Iteration, data: Any, metadata: Dict, **kwargs) -> Any
```

Validate a dictionary of data against the schema.

**Arguments**:

- `data` - The data to validate.
  

**Returns**:

  The validated data.

#### async\_validate(iteration: Iteration, data: Any, metadata: Dict)

```python
async def async_validate(iteration: Iteration, data: Any,
                         metadata: Dict) -> Any
```

Validate a dictionary of data against the schema.

**Arguments**:

- `data` - The data to validate.
  

**Returns**:

  The validated data.

## Schema2Prompt

Class that contains transpilers to go from a schema to its
representation in a prompt.

This is important for communicating the schema to a large language
model, and this class will provide multiple alternatives to do so.

#### datatypes\_to\_xml(dt: DataType, root: Optional[ET.\_Element] = None, override\_tag\_name: Optional[str] = None)

```python
@staticmethod
def datatypes_to_xml(dt: DataType,
                     root: Optional[ET._Element] = None,
                     override_tag_name: Optional[str] = None) -> ET._Element
```

Recursively convert the datatypes to XML elements.

#### default(cls, schema: JsonSchema)

```python
@classmethod
def default(cls, schema: JsonSchema) -> str
```

Default transpiler.

Converts the XML schema to a string directly after removing:
- Comments
- Action attributes like 'on-fail-*'

**Arguments**:

- `schema` - The schema to transpile.
  

**Returns**:

  The prompt.

