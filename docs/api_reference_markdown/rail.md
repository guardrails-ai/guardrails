# Rail

RAIL (Reliable AI Language) is a dialect of XML that allows users to
specify guardrails for large language models (LLMs).

A RAIL file contains a root element called
    `<rail version="x.y">`
that contains the following elements as children:
    1. `<input strict=True/False>`, which contains the input schema
    2. `<output strict=True/False>`, which contains the output schema
    3. `<prompt>`, which contains the prompt to be passed to the LLM
    4. `<instructions>`, which contains the instructions to be passed to the LLM

### from_file `classmethod`

```
from_file(
  file_path: str
) -> Rail
```

### from_pydantic `classmethod`

```
from_pydantic(
  output_class: Type[pydantic.main.BaseModel],
  prompt: Optional[str] = None,
  instructions: Optional[str] = None,
  reask_prompt: Optional[str] = None,
  reask_instructions: Optional[str] = None
)
```

### from_string `classmethod`

```
from_string(
  string: str
) -> Rail
```

### from_string_validators `classmethod`

```
from_string_validators(
  validators: Sequence[Union[guardrails.validator_base.Validator, Tuple[Union[guardrails.validator_base.Validator, str, Callable], str]]],
  description: Optional[str] = None,
  prompt: Optional[str] = None,
  instructions: Optional[str] = None,
  reask_prompt: Optional[str] = None,
  reask_instructions: Optional[str] = None
)
```

### from_xml `classmethod`

```
from_xml(
  xml: lxml.etree._Element
)
```

### output_type `classproperty`

### version `classstr`

str(object='') -> str
str(bytes_or_buffer[, encoding[, errors]]) -> str

Create a new string object from the given object. If encoding or
errors is specified, then the object must expose a data buffer
that will be decoded using the given encoding and error handler.
Otherwise, returns the result of object.__str__() (if defined)
or repr(object).
encoding defaults to sys.getdefaultencoding().
errors defaults to 'strict'.

