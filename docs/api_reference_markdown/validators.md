# Validators

This module contains the validators for the Guardrails framework.

The name with which a validator is registered is the name that is used
in the `RAIL` spec to specify formatters.

## BugFreePython

Validates that there are no Python syntactic bugs in the generated code.

This validator checks for syntax errors by running `ast.parse(code)`,
and will raise an exception if there are any.
Only the packages in the `python` environment are available to the code snippet.

**Key Properties**

| Property                      | Description                       |
| ----------------------------- | --------------------------------- |
| Name for `format` attribute   | `bug-free-python`                 |
| Supported data types          | `string`                          |
| Programmatic fix              | None                              |

### get_args `classfunction`

```
get_args(
  self
)
```

Get the arguments for the validator.

### override_value_on_pass `classbool`

bool(x) -> bool

Returns True when the argument x is true, False otherwise.
The builtins True and False are the only two instances of the class bool.
The class bool is a subclass of the class int, and cannot be subclassed.

### rail_alias `classstr`

str(object='') -> str
str(bytes_or_buffer[, encoding[, errors]]) -> str

Create a new string object from the given object. If encoding or
errors is specified, then the object must expose a data buffer
that will be decoded using the given encoding and error handler.
Otherwise, returns the result of object.__str__() (if defined)
or repr(object).
encoding defaults to sys.getdefaultencoding().
errors defaults to 'strict'.

### required_metadata_keys `classlist`

Built-in mutable sequence.

If no argument is given, the constructor creates a new empty list.
The argument must be an iterable if specified.

### run_in_separate_process `classbool`

bool(x) -> bool

Returns True when the argument x is true, False otherwise.
The builtins True and False are the only two instances of the class bool.
The class bool is a subclass of the class int, and cannot be subclassed.

### to_prompt `classfunction`

```
to_prompt(
  self,
  with_keywords: bool = True
) -> <class 'str'>
```

Convert the validator to a prompt.

E.g. ValidLength(5, 10) -> "length: 5 10" when with_keywords is False.
ValidLength(5, 10) -> "length: min=5 max=10" when with_keywords is True.

Args:
    with_keywords: Whether to include the keyword arguments in the prompt.

Returns:
    A string representation of the validator.

### to_xml_attrib `classfunction`

```
to_xml_attrib(
  self
)
```

Convert the validator to an XML attribute.


## BugFreeSQL

Validates that there are no SQL syntactic bugs in the generated code.

This is a very minimal implementation that uses the Pypi `sqlvalidator` package
to check if the SQL query is valid. You can implement a custom SQL validator
that uses a database connection to check if the query is valid.

**Key Properties**

| Property                      | Description                       |
| ----------------------------- | --------------------------------- |
| Name for `format` attribute   | `bug-free-sql`                    |
| Supported data types          | `string`                          |
| Programmatic fix              | None                              |

### get_args `classfunction`

```
get_args(
  self
)
```

Get the arguments for the validator.

### override_value_on_pass `classbool`

bool(x) -> bool

Returns True when the argument x is true, False otherwise.
The builtins True and False are the only two instances of the class bool.
The class bool is a subclass of the class int, and cannot be subclassed.

### rail_alias `classstr`

str(object='') -> str
str(bytes_or_buffer[, encoding[, errors]]) -> str

Create a new string object from the given object. If encoding or
errors is specified, then the object must expose a data buffer
that will be decoded using the given encoding and error handler.
Otherwise, returns the result of object.__str__() (if defined)
or repr(object).
encoding defaults to sys.getdefaultencoding().
errors defaults to 'strict'.

### required_metadata_keys `classlist`

Built-in mutable sequence.

If no argument is given, the constructor creates a new empty list.
The argument must be an iterable if specified.

### run_in_separate_process `classbool`

bool(x) -> bool

Returns True when the argument x is true, False otherwise.
The builtins True and False are the only two instances of the class bool.
The class bool is a subclass of the class int, and cannot be subclassed.

### to_prompt `classfunction`

```
to_prompt(
  self,
  with_keywords: bool = True
) -> <class 'str'>
```

Convert the validator to a prompt.

E.g. ValidLength(5, 10) -> "length: 5 10" when with_keywords is False.
ValidLength(5, 10) -> "length: min=5 max=10" when with_keywords is True.

Args:
    with_keywords: Whether to include the keyword arguments in the prompt.

Returns:
    A string representation of the validator.

### to_xml_attrib `classfunction`

```
to_xml_attrib(
  self
)
```

Convert the validator to an XML attribute.


## CompetitorCheck

Validates that LLM-generated text is not naming any competitors from a
given list.

In order to use this validator you need to provide an extensive list of the
competitors you want to avoid naming including all common variations.

Args:
    competitors (List[str]): List of competitors you want to avoid naming

### exact_match `classfunction`

```
exact_match(
  self,
  text: str,
  competitors: List[str]
) -> typing.List[str]
```

Performs exact match to find competitors from a list in a given
text.

Args:
    text (str): The text to search for competitors.
    competitors (list): A list of competitor entities to match.

Returns:
    list: A list of matched entities.

### get_args `classfunction`

```
get_args(
  self
)
```

Get the arguments for the validator.

### is_entity_in_list `classfunction`

```
is_entity_in_list(
  self,
  entities: List[str],
  competitors: List[str]
) -> typing.List
```

Checks if any entity from a list is present in a given list of
competitors.

Args:
    entities (list): A list of entities to check
    competitors (list): A list of competitor names to match

Returns:
    List: List of found competitors

### override_value_on_pass `classbool`

bool(x) -> bool

Returns True when the argument x is true, False otherwise.
The builtins True and False are the only two instances of the class bool.
The class bool is a subclass of the class int, and cannot be subclassed.

### perform_ner `classfunction`

```
perform_ner(
  self,
  text: str,
  nlp
) -> typing.List[str]
```

Performs named entity recognition on text using a provided NLP
model.

Args:
    text (str): The text to perform named entity recognition on.
    nlp: The NLP model to use for entity recognition.

Returns:
    entities: A list of entities found.

### rail_alias `classstr`

str(object='') -> str
str(bytes_or_buffer[, encoding[, errors]]) -> str

Create a new string object from the given object. If encoding or
errors is specified, then the object must expose a data buffer
that will be decoded using the given encoding and error handler.
Otherwise, returns the result of object.__str__() (if defined)
or repr(object).
encoding defaults to sys.getdefaultencoding().
errors defaults to 'strict'.

### required_metadata_keys `classlist`

Built-in mutable sequence.

If no argument is given, the constructor creates a new empty list.
The argument must be an iterable if specified.

### run_in_separate_process `classbool`

bool(x) -> bool

Returns True when the argument x is true, False otherwise.
The builtins True and False are the only two instances of the class bool.
The class bool is a subclass of the class int, and cannot be subclassed.

### to_prompt `classfunction`

```
to_prompt(
  self,
  with_keywords: bool = True
) -> <class 'str'>
```

Convert the validator to a prompt.

E.g. ValidLength(5, 10) -> "length: 5 10" when with_keywords is False.
ValidLength(5, 10) -> "length: min=5 max=10" when with_keywords is True.

Args:
    with_keywords: Whether to include the keyword arguments in the prompt.

Returns:
    A string representation of the validator.

### to_xml_attrib `classfunction`

```
to_xml_attrib(
  self
)
```

Convert the validator to an XML attribute.


## DetectSecrets

Validates whether the generated code snippet contains any secrets.

**Key Properties**
| Property                      | Description                       |
| ----------------------------- | --------------------------------- |
| Name for `format` attribute   | `detect-secrets`                  |
| Supported data types          | `string`                          |
| Programmatic fix              | None                              |

Parameters: Arguments
    None

This validator uses the detect-secrets library to check whether the generated code
snippet contains any secrets. If any secrets are detected, the validator fails and
returns the generated code snippet with the secrets replaced with asterisks.
Else the validator returns the generated code snippet.

Following are some caveats:
    - Multiple secrets on the same line may not be caught. e.g.
        - Minified code
        - One-line lists/dictionaries
        - Multi-variable assignments
    - Multi-line secrets may not be caught. e.g.
        - RSA/SSH keys

Example:
    ```py

    guard = Guard.from_string(validators=[
        DetectSecrets(on_fail="fix")
    ])
    guard.parse(
        llm_output=code_snippet,
    )

### get_args `classfunction`

```
get_args(
  self
)
```

Get the arguments for the validator.

### get_modified_value `classfunction`

```
get_modified_value(
  self,
  unique_secrets: Dict[str, Any],
  lines: List[str]
) -> <class 'str'>
```

Replace the secrets on the lines with asterisks.

Args:
    unique_secrets (Dict[str, Any]): A dictionary of unique secrets and their
        line numbers.
    lines (List[str]): The lines of the generated code snippet.

Returns:
    modified_value (str): The generated code snippet with secrets replaced with
        asterisks.

### get_unique_secrets `classfunction`

```
get_unique_secrets(
  self,
  value: str
) -> typing.Tuple[typing.Dict[str, typing.Any], typing.List[str]]
```

Get unique secrets from the value.

Args:
    value (str): The generated code snippet.

Returns:
    unique_secrets (Dict[str, Any]): A dictionary of unique secrets and their
        line numbers.
    lines (List[str]): The lines of the generated code snippet.

### override_value_on_pass `classbool`

bool(x) -> bool

Returns True when the argument x is true, False otherwise.
The builtins True and False are the only two instances of the class bool.
The class bool is a subclass of the class int, and cannot be subclassed.

### rail_alias `classstr`

str(object='') -> str
str(bytes_or_buffer[, encoding[, errors]]) -> str

Create a new string object from the given object. If encoding or
errors is specified, then the object must expose a data buffer
that will be decoded using the given encoding and error handler.
Otherwise, returns the result of object.__str__() (if defined)
or repr(object).
encoding defaults to sys.getdefaultencoding().
errors defaults to 'strict'.

### required_metadata_keys `classlist`

Built-in mutable sequence.

If no argument is given, the constructor creates a new empty list.
The argument must be an iterable if specified.

### run_in_separate_process `classbool`

bool(x) -> bool

Returns True when the argument x is true, False otherwise.
The builtins True and False are the only two instances of the class bool.
The class bool is a subclass of the class int, and cannot be subclassed.

### to_prompt `classfunction`

```
to_prompt(
  self,
  with_keywords: bool = True
) -> <class 'str'>
```

Convert the validator to a prompt.

E.g. ValidLength(5, 10) -> "length: 5 10" when with_keywords is False.
ValidLength(5, 10) -> "length: min=5 max=10" when with_keywords is True.

Args:
    with_keywords: Whether to include the keyword arguments in the prompt.

Returns:
    A string representation of the validator.

### to_xml_attrib `classfunction`

```
to_xml_attrib(
  self
)
```

Convert the validator to an XML attribute.


## EndpointIsReachable

Validates that a value is a reachable URL.

**Key Properties**

| Property                      | Description                       |
| ----------------------------- | --------------------------------- |
| Name for `format` attribute   | `is-reachable`                    |
| Supported data types          | `string`,                         |
| Programmatic fix              | None                              |

### get_args `classfunction`

```
get_args(
  self
)
```

Get the arguments for the validator.

### override_value_on_pass `classbool`

bool(x) -> bool

Returns True when the argument x is true, False otherwise.
The builtins True and False are the only two instances of the class bool.
The class bool is a subclass of the class int, and cannot be subclassed.

### rail_alias `classstr`

str(object='') -> str
str(bytes_or_buffer[, encoding[, errors]]) -> str

Create a new string object from the given object. If encoding or
errors is specified, then the object must expose a data buffer
that will be decoded using the given encoding and error handler.
Otherwise, returns the result of object.__str__() (if defined)
or repr(object).
encoding defaults to sys.getdefaultencoding().
errors defaults to 'strict'.

### required_metadata_keys `classlist`

Built-in mutable sequence.

If no argument is given, the constructor creates a new empty list.
The argument must be an iterable if specified.

### run_in_separate_process `classbool`

bool(x) -> bool

Returns True when the argument x is true, False otherwise.
The builtins True and False are the only two instances of the class bool.
The class bool is a subclass of the class int, and cannot be subclassed.

### to_prompt `classfunction`

```
to_prompt(
  self,
  with_keywords: bool = True
) -> <class 'str'>
```

Convert the validator to a prompt.

E.g. ValidLength(5, 10) -> "length: 5 10" when with_keywords is False.
ValidLength(5, 10) -> "length: min=5 max=10" when with_keywords is True.

Args:
    with_keywords: Whether to include the keyword arguments in the prompt.

Returns:
    A string representation of the validator.

### to_xml_attrib `classfunction`

```
to_xml_attrib(
  self
)
```

Convert the validator to an XML attribute.


## EndsWith

Validates that a list ends with a given value.

**Key Properties**

| Property                      | Description                       |
| ----------------------------- | --------------------------------- |
| Name for `format` attribute   | `ends-with`                       |
| Supported data types          | `list`                            |
| Programmatic fix              | Append the given value to the list. |

Parameters: Arguments
    end: The required last element.

### get_args `classfunction`

```
get_args(
  self
)
```

Get the arguments for the validator.

### override_value_on_pass `classbool`

bool(x) -> bool

Returns True when the argument x is true, False otherwise.
The builtins True and False are the only two instances of the class bool.
The class bool is a subclass of the class int, and cannot be subclassed.

### rail_alias `classstr`

str(object='') -> str
str(bytes_or_buffer[, encoding[, errors]]) -> str

Create a new string object from the given object. If encoding or
errors is specified, then the object must expose a data buffer
that will be decoded using the given encoding and error handler.
Otherwise, returns the result of object.__str__() (if defined)
or repr(object).
encoding defaults to sys.getdefaultencoding().
errors defaults to 'strict'.

### required_metadata_keys `classlist`

Built-in mutable sequence.

If no argument is given, the constructor creates a new empty list.
The argument must be an iterable if specified.

### run_in_separate_process `classbool`

bool(x) -> bool

Returns True when the argument x is true, False otherwise.
The builtins True and False are the only two instances of the class bool.
The class bool is a subclass of the class int, and cannot be subclassed.

### to_prompt `classfunction`

```
to_prompt(
  self,
  with_keywords: bool = True
) -> <class 'str'>
```

Convert the validator to a prompt.

E.g. ValidLength(5, 10) -> "length: 5 10" when with_keywords is False.
ValidLength(5, 10) -> "length: min=5 max=10" when with_keywords is True.

Args:
    with_keywords: Whether to include the keyword arguments in the prompt.

Returns:
    A string representation of the validator.

### to_xml_attrib `classfunction`

```
to_xml_attrib(
  self
)
```

Convert the validator to an XML attribute.


## ExcludeSqlPredicates

Validates that the SQL query does not contain certain predicates.

**Key Properties**

| Property                      | Description                       |
| ----------------------------- | --------------------------------- |
| Name for `format` attribute   | `exclude-sql-predicates`          |
| Supported data types          | `string`                          |
| Programmatic fix              | None                              |

Parameters: Arguments
    predicates: The list of predicates to avoid.

### get_args `classfunction`

```
get_args(
  self
)
```

Get the arguments for the validator.

### override_value_on_pass `classbool`

bool(x) -> bool

Returns True when the argument x is true, False otherwise.
The builtins True and False are the only two instances of the class bool.
The class bool is a subclass of the class int, and cannot be subclassed.

### rail_alias `classstr`

str(object='') -> str
str(bytes_or_buffer[, encoding[, errors]]) -> str

Create a new string object from the given object. If encoding or
errors is specified, then the object must expose a data buffer
that will be decoded using the given encoding and error handler.
Otherwise, returns the result of object.__str__() (if defined)
or repr(object).
encoding defaults to sys.getdefaultencoding().
errors defaults to 'strict'.

### required_metadata_keys `classlist`

Built-in mutable sequence.

If no argument is given, the constructor creates a new empty list.
The argument must be an iterable if specified.

### run_in_separate_process `classbool`

bool(x) -> bool

Returns True when the argument x is true, False otherwise.
The builtins True and False are the only two instances of the class bool.
The class bool is a subclass of the class int, and cannot be subclassed.

### to_prompt `classfunction`

```
to_prompt(
  self,
  with_keywords: bool = True
) -> <class 'str'>
```

Convert the validator to a prompt.

E.g. ValidLength(5, 10) -> "length: 5 10" when with_keywords is False.
ValidLength(5, 10) -> "length: min=5 max=10" when with_keywords is True.

Args:
    with_keywords: Whether to include the keyword arguments in the prompt.

Returns:
    A string representation of the validator.

### to_xml_attrib `classfunction`

```
to_xml_attrib(
  self
)
```

Convert the validator to an XML attribute.


## ExtractedSummarySentencesMatch

Validates that the extracted summary sentences match the original text
by performing a cosine similarity in the embedding space.

**Key Properties**

| Property                      | Description                         |
| ----------------------------- | ----------------------------------- |
| Name for `format` attribute   | `extracted-summary-sentences-match` |
| Supported data types          | `string`                            |
| Programmatic fix              | Remove any sentences that can not be verified. |

Parameters: Arguments

    threshold: The minimum cosine similarity to be considered similar. Default to 0.7.

Other parameters: Metadata

    filepaths (List[str]): A list of strings that specifies the filepaths for any documents that should be used for asserting the summary's similarity.
    document_store (DocumentStoreBase, optional): The document store to use during validation. Defaults to EphemeralDocumentStore.
    vector_db (VectorDBBase, optional): A vector database to use for embeddings.  Defaults to Faiss.
    embedding_model (EmbeddingBase, optional): The embeddig model to use. Defaults to OpenAIEmbedding.

### _instantiate_store `classfunction`

```
_instantiate_store(
  metadata,
  api_key: Optional[str] = None,
  api_base: Optional[str] = None
)
```

### get_args `classfunction`

```
get_args(
  self
)
```

Get the arguments for the validator.

### override_value_on_pass `classbool`

bool(x) -> bool

Returns True when the argument x is true, False otherwise.
The builtins True and False are the only two instances of the class bool.
The class bool is a subclass of the class int, and cannot be subclassed.

### rail_alias `classstr`

str(object='') -> str
str(bytes_or_buffer[, encoding[, errors]]) -> str

Create a new string object from the given object. If encoding or
errors is specified, then the object must expose a data buffer
that will be decoded using the given encoding and error handler.
Otherwise, returns the result of object.__str__() (if defined)
or repr(object).
encoding defaults to sys.getdefaultencoding().
errors defaults to 'strict'.

### required_metadata_keys `classlist`

Built-in mutable sequence.

If no argument is given, the constructor creates a new empty list.
The argument must be an iterable if specified.

### run_in_separate_process `classbool`

bool(x) -> bool

Returns True when the argument x is true, False otherwise.
The builtins True and False are the only two instances of the class bool.
The class bool is a subclass of the class int, and cannot be subclassed.

### to_prompt `classfunction`

```
to_prompt(
  self,
  with_keywords: bool = True
) -> <class 'str'>
```

Convert the validator to a prompt.

E.g. ValidLength(5, 10) -> "length: 5 10" when with_keywords is False.
ValidLength(5, 10) -> "length: min=5 max=10" when with_keywords is True.

Args:
    with_keywords: Whether to include the keyword arguments in the prompt.

Returns:
    A string representation of the validator.

### to_xml_attrib `classfunction`

```
to_xml_attrib(
  self
)
```

Convert the validator to an XML attribute.


## ExtractiveSummary

Validates that a string is a valid extractive summary of a given
document.

This validator does a fuzzy match between the sentences in the
summary and the sentences in the document. Each sentence in the
summary must be similar to at least one sentence in the document.
After the validation, the summary is updated to include the
sentences from the document that were matched, and the citations for
those sentences are added to the end of the summary.

**Key Properties**

| Property                      | Description                         |
| ----------------------------- | ----------------------------------- |
| Name for `format` attribute   | `extractive-summary`                |
| Supported data types          | `string`                            |
| Programmatic fix              | Remove any sentences that can not be verified. |

Parameters: Arguments

    threshold: The minimum fuzz ratio to be considered summarized.  Defaults to 85.

Other parameters: Metadata

    filepaths (List[str]): A list of strings that specifies the filepaths for any documents that should be used for asserting the summary's similarity.

### get_args `classfunction`

```
get_args(
  self
)
```

Get the arguments for the validator.

### override_value_on_pass `classbool`

bool(x) -> bool

Returns True when the argument x is true, False otherwise.
The builtins True and False are the only two instances of the class bool.
The class bool is a subclass of the class int, and cannot be subclassed.

### rail_alias `classstr`

str(object='') -> str
str(bytes_or_buffer[, encoding[, errors]]) -> str

Create a new string object from the given object. If encoding or
errors is specified, then the object must expose a data buffer
that will be decoded using the given encoding and error handler.
Otherwise, returns the result of object.__str__() (if defined)
or repr(object).
encoding defaults to sys.getdefaultencoding().
errors defaults to 'strict'.

### required_metadata_keys `classlist`

Built-in mutable sequence.

If no argument is given, the constructor creates a new empty list.
The argument must be an iterable if specified.

### run_in_separate_process `classbool`

bool(x) -> bool

Returns True when the argument x is true, False otherwise.
The builtins True and False are the only two instances of the class bool.
The class bool is a subclass of the class int, and cannot be subclassed.

### to_prompt `classfunction`

```
to_prompt(
  self,
  with_keywords: bool = True
) -> <class 'str'>
```

Convert the validator to a prompt.

E.g. ValidLength(5, 10) -> "length: 5 10" when with_keywords is False.
ValidLength(5, 10) -> "length: min=5 max=10" when with_keywords is True.

Args:
    with_keywords: Whether to include the keyword arguments in the prompt.

Returns:
    A string representation of the validator.

### to_xml_attrib `classfunction`

```
to_xml_attrib(
  self
)
```

Convert the validator to an XML attribute.


## IsHighQualityTranslation

Validates that the translation is of high quality.

**Key Properties**

| Property                      | Description                       |
| ----------------------------- | --------------------------------- |
| Name for `format` attribute   | `is-high-quality-translation`     |
| Supported data types          | `string`                          |
| Programmatic fix              | None                              |

Other parameters: Metadata
    translation_source (str): The source of the translation.

This validator uses one of the reference-free models from Unbabel/COMET
to check the quality of the translation. Specifically, it uses the
`Unbabel/wmt22-cometkiwi-da` model.

Unbabel/COMET details: https://github.com/Unbabel/COMET
Model details: https://huggingface.co/Unbabel/wmt22-cometkiwi-da

Pre-requisites:
    - Install the `unbabel-comet` from source:
        `pip install git+https://github.com/Unbabel/COMET`
    - Please accept the model license from:
        https://huggingface.co/Unbabel/wmt22-cometkiwi-da
    - Login into Huggingface Hub using:
        huggingface-cli login --token $HUGGINGFACE_TOKEN

### get_args `classfunction`

```
get_args(
  self
)
```

Get the arguments for the validator.

### override_value_on_pass `classbool`

bool(x) -> bool

Returns True when the argument x is true, False otherwise.
The builtins True and False are the only two instances of the class bool.
The class bool is a subclass of the class int, and cannot be subclassed.

### rail_alias `classstr`

str(object='') -> str
str(bytes_or_buffer[, encoding[, errors]]) -> str

Create a new string object from the given object. If encoding or
errors is specified, then the object must expose a data buffer
that will be decoded using the given encoding and error handler.
Otherwise, returns the result of object.__str__() (if defined)
or repr(object).
encoding defaults to sys.getdefaultencoding().
errors defaults to 'strict'.

### required_metadata_keys `classlist`

Built-in mutable sequence.

If no argument is given, the constructor creates a new empty list.
The argument must be an iterable if specified.

### run_in_separate_process `classbool`

bool(x) -> bool

Returns True when the argument x is true, False otherwise.
The builtins True and False are the only two instances of the class bool.
The class bool is a subclass of the class int, and cannot be subclassed.

### to_prompt `classfunction`

```
to_prompt(
  self,
  with_keywords: bool = True
) -> <class 'str'>
```

Convert the validator to a prompt.

E.g. ValidLength(5, 10) -> "length: 5 10" when with_keywords is False.
ValidLength(5, 10) -> "length: min=5 max=10" when with_keywords is True.

Args:
    with_keywords: Whether to include the keyword arguments in the prompt.

Returns:
    A string representation of the validator.

### to_xml_attrib `classfunction`

```
to_xml_attrib(
  self
)
```

Convert the validator to an XML attribute.


## IsProfanityFree

Validates that a translated text does not contain profanity language.

This validator uses the `alt-profanity-check` package to check if a string
contains profanity language.

**Key Properties**

| Property                      | Description                       |
| ----------------------------- | --------------------------------- |
| Name for `format` attribute   | `is-profanity-free`               |
| Supported data types          | `string`                          |
| Programmatic fix              | None                              |

### get_args `classfunction`

```
get_args(
  self
)
```

Get the arguments for the validator.

### override_value_on_pass `classbool`

bool(x) -> bool

Returns True when the argument x is true, False otherwise.
The builtins True and False are the only two instances of the class bool.
The class bool is a subclass of the class int, and cannot be subclassed.

### rail_alias `classstr`

str(object='') -> str
str(bytes_or_buffer[, encoding[, errors]]) -> str

Create a new string object from the given object. If encoding or
errors is specified, then the object must expose a data buffer
that will be decoded using the given encoding and error handler.
Otherwise, returns the result of object.__str__() (if defined)
or repr(object).
encoding defaults to sys.getdefaultencoding().
errors defaults to 'strict'.

### required_metadata_keys `classlist`

Built-in mutable sequence.

If no argument is given, the constructor creates a new empty list.
The argument must be an iterable if specified.

### run_in_separate_process `classbool`

bool(x) -> bool

Returns True when the argument x is true, False otherwise.
The builtins True and False are the only two instances of the class bool.
The class bool is a subclass of the class int, and cannot be subclassed.

### to_prompt `classfunction`

```
to_prompt(
  self,
  with_keywords: bool = True
) -> <class 'str'>
```

Convert the validator to a prompt.

E.g. ValidLength(5, 10) -> "length: 5 10" when with_keywords is False.
ValidLength(5, 10) -> "length: min=5 max=10" when with_keywords is True.

Args:
    with_keywords: Whether to include the keyword arguments in the prompt.

Returns:
    A string representation of the validator.

### to_xml_attrib `classfunction`

```
to_xml_attrib(
  self
)
```

Convert the validator to an XML attribute.


## LowerCase

Validates that a value is lower case.

**Key Properties**

| Property                      | Description                       |
| ----------------------------- | --------------------------------- |
| Name for `format` attribute   | `lower-case`                      |
| Supported data types          | `string`                          |
| Programmatic fix              | Convert to lower case.            |

### get_args `classfunction`

```
get_args(
  self
)
```

Get the arguments for the validator.

### override_value_on_pass `classbool`

bool(x) -> bool

Returns True when the argument x is true, False otherwise.
The builtins True and False are the only two instances of the class bool.
The class bool is a subclass of the class int, and cannot be subclassed.

### rail_alias `classstr`

str(object='') -> str
str(bytes_or_buffer[, encoding[, errors]]) -> str

Create a new string object from the given object. If encoding or
errors is specified, then the object must expose a data buffer
that will be decoded using the given encoding and error handler.
Otherwise, returns the result of object.__str__() (if defined)
or repr(object).
encoding defaults to sys.getdefaultencoding().
errors defaults to 'strict'.

### required_metadata_keys `classlist`

Built-in mutable sequence.

If no argument is given, the constructor creates a new empty list.
The argument must be an iterable if specified.

### run_in_separate_process `classbool`

bool(x) -> bool

Returns True when the argument x is true, False otherwise.
The builtins True and False are the only two instances of the class bool.
The class bool is a subclass of the class int, and cannot be subclassed.

### to_prompt `classfunction`

```
to_prompt(
  self,
  with_keywords: bool = True
) -> <class 'str'>
```

Convert the validator to a prompt.

E.g. ValidLength(5, 10) -> "length: 5 10" when with_keywords is False.
ValidLength(5, 10) -> "length: min=5 max=10" when with_keywords is True.

Args:
    with_keywords: Whether to include the keyword arguments in the prompt.

Returns:
    A string representation of the validator.

### to_xml_attrib `classfunction`

```
to_xml_attrib(
  self
)
```

Convert the validator to an XML attribute.


## OnTopic

Checks if text's main topic is specified within a list of valid topics
and ensures that the text is not about any of the invalid topics.

This validator accepts at least one valid topic and an optional list of
invalid topics.

Default behavior first runs a Zero-Shot model, and then falls back to
ask OpenAI's `gpt-3.5-turbo` if the Zero-Shot model is not confident
in the topic classification (score < 0.5).

In our experiments this LLM fallback increases accuracy by 15% but also
increases latency (more than doubles the latency in the worst case).

Both the Zero-Shot classification and the GPT classification may be toggled.

**Key Properties**

| Property                      | Description                              |
| ----------------------------- | ---------------------------------------- |
| Name for `format` attribute   | `on_topic`                               |
| Supported data types          | `string`                                 |
| Programmatic fix              | Removes lines with off-topic information |

Parameters: Arguments
    valid_topics (List[str]): topics that the text should be about
        (one or many).
    invalid_topics (List[str], Optional, defaults to []): topics that the
        text cannot be about.
    device (int, Optional, defaults to -1): Device ordinal for CPU/GPU
        supports for Zero-Shot classifier. Setting this to -1 will leverage
        CPU, a positive will run the Zero-Shot model on the associated CUDA
        device id.
    model (str, Optional, defaults to 'facebook/bart-large-mnli'): The
        Zero-Shot model that will be used to classify the topic. See a
        list of all models here:
        https://huggingface.co/models?pipeline_tag=zero-shot-classification
    llm_callable (Union[str, Callable, None], Optional, defaults to
        'gpt-3.5-turbo'): Either the name of the OpenAI model, or a callable
        that takes a prompt and returns a response.
    disable_classifier (bool, Optional, defaults to False): controls whether
        to use the Zero-Shot model. At least one of disable_classifier and
        disable_llm must be False.
    disable_llm (bool, Optional, defaults to False): controls whether to use
        the LLM fallback. At least one of disable_classifier and
        disable_llm must be False.
    model_threshold (float, Optional, defaults to 0.5): The threshold used to
        determine whether to accept a topic from the Zero-Shot model. Must be
        a number between 0 and 1.

### call_llm `classfunction`

```
call_llm(
  self,
  text: str,
  topics: List[str]
) -> <class 'str'>
```

Call the LLM with the given prompt.

Expects a function that takes a string and returns a string.
Args:
    text (str): The input text to classify using the LLM.
    topics (List[str]): The list of candidate topics.
Returns:
    response (str): String representing the LLM response.

### get_args `classfunction`

```
get_args(
  self
)
```

Get the arguments for the validator.

### get_client_args `classfunction`

```
get_client_args(
  self
) -> typing.Tuple[typing.Optional[str], typing.Optional[str]]
```

### get_topic_ensemble `classfunction`

```
get_topic_ensemble(
  self,
  text: str,
  candidate_topics: List[str]
) -> <class 'guardrails.validator_base.ValidationResult'>
```

### get_topic_llm `classfunction`

```
get_topic_llm(
  self,
  text: str,
  candidate_topics: List[str]
) -> <class 'guardrails.validator_base.ValidationResult'>
```

### get_topic_zero_shot `classfunction`

```
get_topic_zero_shot(
  self,
  text: str,
  candidate_topics: List[str]
) -> typing.Tuple[str, float]
```

### override_value_on_pass `classbool`

bool(x) -> bool

Returns True when the argument x is true, False otherwise.
The builtins True and False are the only two instances of the class bool.
The class bool is a subclass of the class int, and cannot be subclassed.

### rail_alias `classstr`

str(object='') -> str
str(bytes_or_buffer[, encoding[, errors]]) -> str

Create a new string object from the given object. If encoding or
errors is specified, then the object must expose a data buffer
that will be decoded using the given encoding and error handler.
Otherwise, returns the result of object.__str__() (if defined)
or repr(object).
encoding defaults to sys.getdefaultencoding().
errors defaults to 'strict'.

### required_metadata_keys `classlist`

Built-in mutable sequence.

If no argument is given, the constructor creates a new empty list.
The argument must be an iterable if specified.

### run_in_separate_process `classbool`

bool(x) -> bool

Returns True when the argument x is true, False otherwise.
The builtins True and False are the only two instances of the class bool.
The class bool is a subclass of the class int, and cannot be subclassed.

### set_callable `classfunction`

```
set_callable(
  self,
  llm_callable: Union[str, Callable, NoneType]
) -> None
```

Set the LLM callable.

Args:
    llm_callable: Either the name of the OpenAI model, or a callable that takes
        a prompt and returns a response.

### to_prompt `classfunction`

```
to_prompt(
  self,
  with_keywords: bool = True
) -> <class 'str'>
```

Convert the validator to a prompt.

E.g. ValidLength(5, 10) -> "length: 5 10" when with_keywords is False.
ValidLength(5, 10) -> "length: min=5 max=10" when with_keywords is True.

Args:
    with_keywords: Whether to include the keyword arguments in the prompt.

Returns:
    A string representation of the validator.

### to_xml_attrib `classfunction`

```
to_xml_attrib(
  self
)
```

Convert the validator to an XML attribute.

### verify_topic `classfunction`

```
verify_topic(
  self,
  topic: str
) -> <class 'guardrails.validator_base.ValidationResult'>
```


## OneLine

Validates that a value is a single line, based on whether or not the
output has a newline character (\n).

**Key Properties**

| Property                      | Description                            |
| ----------------------------- | -------------------------------------- |
| Name for `format` attribute   | `one-line`                             |
| Supported data types          | `string`                               |
| Programmatic fix              | Keep the first line, delete other text |

### get_args `classfunction`

```
get_args(
  self
)
```

Get the arguments for the validator.

### override_value_on_pass `classbool`

bool(x) -> bool

Returns True when the argument x is true, False otherwise.
The builtins True and False are the only two instances of the class bool.
The class bool is a subclass of the class int, and cannot be subclassed.

### rail_alias `classstr`

str(object='') -> str
str(bytes_or_buffer[, encoding[, errors]]) -> str

Create a new string object from the given object. If encoding or
errors is specified, then the object must expose a data buffer
that will be decoded using the given encoding and error handler.
Otherwise, returns the result of object.__str__() (if defined)
or repr(object).
encoding defaults to sys.getdefaultencoding().
errors defaults to 'strict'.

### required_metadata_keys `classlist`

Built-in mutable sequence.

If no argument is given, the constructor creates a new empty list.
The argument must be an iterable if specified.

### run_in_separate_process `classbool`

bool(x) -> bool

Returns True when the argument x is true, False otherwise.
The builtins True and False are the only two instances of the class bool.
The class bool is a subclass of the class int, and cannot be subclassed.

### to_prompt `classfunction`

```
to_prompt(
  self,
  with_keywords: bool = True
) -> <class 'str'>
```

Convert the validator to a prompt.

E.g. ValidLength(5, 10) -> "length: 5 10" when with_keywords is False.
ValidLength(5, 10) -> "length: min=5 max=10" when with_keywords is True.

Args:
    with_keywords: Whether to include the keyword arguments in the prompt.

Returns:
    A string representation of the validator.

### to_xml_attrib `classfunction`

```
to_xml_attrib(
  self
)
```

Convert the validator to an XML attribute.


## PIIFilter

Validates that any text does not contain any PII.

This validator uses Microsoft's Presidio (https://github.com/microsoft/presidio)
to detect PII in the text. If PII is detected, the validator will fail with a
programmatic fix that anonymizes the text. Otherwise, the validator will pass.

**Key Properties**

| Property                      | Description                         |
| ----------------------------- | ----------------------------------- |
| Name for `format` attribute   | `pii`                               |
| Supported data types          | `string`                            |
| Programmatic fix              | Anonymized text with PII filtered   |

Parameters: Arguments
    pii_entities (str | List[str], optional): The PII entities to filter. Must be
        one of `pii` or `spi`. Defaults to None. Can also be set in metadata.

### PII_ENTITIES_MAP `classdict`

dict() -> new empty dictionary
dict(mapping) -> new dictionary initialized from a mapping object's
    (key, value) pairs
dict(iterable) -> new dictionary initialized as if via:
    d = {}
    for k, v in iterable:
        d[k] = v
dict(**kwargs) -> new dictionary initialized with the name=value pairs
    in the keyword argument list.  For example:  dict(one=1, two=2)

### get_anonymized_text `classfunction`

```
get_anonymized_text(
  self,
  text: str,
  entities: List[str]
) -> <class 'str'>
```

Analyze and anonymize the text for PII.

Args:
    text (str): The text to analyze.
    pii_entities (List[str]): The PII entities to filter.

Returns:
    anonymized_text (str): The anonymized text.

### get_args `classfunction`

```
get_args(
  self
)
```

Get the arguments for the validator.

### override_value_on_pass `classbool`

bool(x) -> bool

Returns True when the argument x is true, False otherwise.
The builtins True and False are the only two instances of the class bool.
The class bool is a subclass of the class int, and cannot be subclassed.

### rail_alias `classstr`

str(object='') -> str
str(bytes_or_buffer[, encoding[, errors]]) -> str

Create a new string object from the given object. If encoding or
errors is specified, then the object must expose a data buffer
that will be decoded using the given encoding and error handler.
Otherwise, returns the result of object.__str__() (if defined)
or repr(object).
encoding defaults to sys.getdefaultencoding().
errors defaults to 'strict'.

### required_metadata_keys `classlist`

Built-in mutable sequence.

If no argument is given, the constructor creates a new empty list.
The argument must be an iterable if specified.

### run_in_separate_process `classbool`

bool(x) -> bool

Returns True when the argument x is true, False otherwise.
The builtins True and False are the only two instances of the class bool.
The class bool is a subclass of the class int, and cannot be subclassed.

### to_prompt `classfunction`

```
to_prompt(
  self,
  with_keywords: bool = True
) -> <class 'str'>
```

Convert the validator to a prompt.

E.g. ValidLength(5, 10) -> "length: 5 10" when with_keywords is False.
ValidLength(5, 10) -> "length: min=5 max=10" when with_keywords is True.

Args:
    with_keywords: Whether to include the keyword arguments in the prompt.

Returns:
    A string representation of the validator.

### to_xml_attrib `classfunction`

```
to_xml_attrib(
  self
)
```

Convert the validator to an XML attribute.


## ProvenanceV0

Validates that LLM-generated text matches some source text based on
distance in embedding space.

**Key Properties**

| Property                      | Description                         |
| ----------------------------- | ----------------------------------- |
| Name for `format` attribute   | `provenance-v0`                     |
| Supported data types          | `string`                            |
| Programmatic fix              | None                                |

Parameters: Arguments
    threshold: The minimum cosine similarity between the generated text and
        the source text. Defaults to 0.8.
    validation_method: Whether to validate at the sentence level or over the full text.  Must be one of `sentence` or `full`. Defaults to `sentence`

Other parameters: Metadata
    query_function (Callable, optional): A callable that takes a string and returns a list of (chunk, score) tuples.
    sources (List[str], optional): The source text.
    embed_function (Callable, optional): A callable that creates embeddings for the sources. Must accept a list of strings and return an np.array of floats.

In order to use this validator, you must provide either a `query_function` or
`sources` with an `embed_function` in the metadata.

If providing query_function, it should take a string as input and return a list of
(chunk, score) tuples. The chunk is a string and the score is a float representing
the cosine distance between the chunk and the input string. The list should be
sorted in ascending order by score.

Note: The score should represent distance in embedding space, not similarity. I.e.,
lower is better and the score should be 0 if the chunk is identical to the input
string.

Example:
    ```py
    def query_function(text: str, k: int) -> List[Tuple[str, float]]:
        return [("This is a chunk", 0.9), ("This is another chunk", 0.8)]

    guard = Guard.from_rail(...)
    guard(
        openai.ChatCompletion.create(...),
        prompt_params={...},
        temperature=0.0,
        metadata={"query_function": query_function},
    )
    ```


If providing sources, it should be a list of strings. The embed_function should
take a string or a list of strings as input and return a np array of floats.
The vector should be normalized to unit length.

Example:
    ```py
    def embed_function(text: Union[str, List[str]]) -> np.ndarray:
        return np.array([[0.1, 0.2, 0.3]])

    guard = Guard.from_rail(...)
    guard(
        openai.ChatCompletion.create(...),
        prompt_params={...},
        temperature=0.0,
        metadata={
            "sources": ["This is a source text"],
            "embed_function": embed_function
        },
    )
    ```

### get_args `classfunction`

```
get_args(
  self
)
```

Get the arguments for the validator.

### get_query_function `classfunction`

```
get_query_function(
  self,
  metadata: Dict[str, Any]
) -> typing.Callable
```

### override_value_on_pass `classbool`

bool(x) -> bool

Returns True when the argument x is true, False otherwise.
The builtins True and False are the only two instances of the class bool.
The class bool is a subclass of the class int, and cannot be subclassed.

### query_vector_collection `classfunction`

```
query_vector_collection(
  text: str,
  k: int,
  sources: List[str],
  embed_function: Callable,
  chunk_strategy: str = 'sentence',
  chunk_size: int = 5,
  chunk_overlap: int = 2,
  distance_metric: str = 'cosine'
) -> typing.List[typing.Tuple[str, float]]
```

### rail_alias `classstr`

str(object='') -> str
str(bytes_or_buffer[, encoding[, errors]]) -> str

Create a new string object from the given object. If encoding or
errors is specified, then the object must expose a data buffer
that will be decoded using the given encoding and error handler.
Otherwise, returns the result of object.__str__() (if defined)
or repr(object).
encoding defaults to sys.getdefaultencoding().
errors defaults to 'strict'.

### required_metadata_keys `classlist`

Built-in mutable sequence.

If no argument is given, the constructor creates a new empty list.
The argument must be an iterable if specified.

### run_in_separate_process `classbool`

bool(x) -> bool

Returns True when the argument x is true, False otherwise.
The builtins True and False are the only two instances of the class bool.
The class bool is a subclass of the class int, and cannot be subclassed.

### to_prompt `classfunction`

```
to_prompt(
  self,
  with_keywords: bool = True
) -> <class 'str'>
```

Convert the validator to a prompt.

E.g. ValidLength(5, 10) -> "length: 5 10" when with_keywords is False.
ValidLength(5, 10) -> "length: min=5 max=10" when with_keywords is True.

Args:
    with_keywords: Whether to include the keyword arguments in the prompt.

Returns:
    A string representation of the validator.

### to_xml_attrib `classfunction`

```
to_xml_attrib(
  self
)
```

Convert the validator to an XML attribute.


## ProvenanceV1

Validates that the LLM-generated text is supported by the provided
contexts.

This validator uses an LLM callable to evaluate the generated text against the
provided contexts (LLM-ception).

In order to use this validator, you must provide either:
1. a 'query_function' in the metadata. That function should take a string as input
    (the LLM-generated text) and return a list of relevant
chunks. The list should be sorted in ascending order by the distance between the
    chunk and the LLM-generated text.

Example using str callable:
    >>> def query_function(text: str, k: int) -> List[str]:
    ...     return ["This is a chunk", "This is another chunk"]

    >>> guard = Guard.from_string(validators=[
                ProvenanceV1(llm_callable="gpt-3.5-turbo", ...)
            ]
        )
    >>> guard.parse(
    ...   llm_output=...,
    ...   metadata={"query_function": query_function}
    ... )

Example using a custom llm callable:
    >>> def query_function(text: str, k: int) -> List[str]:
    ...     return ["This is a chunk", "This is another chunk"]

    >>> guard = Guard.from_string(validators=[
                ProvenanceV1(llm_callable=your_custom_callable, ...)
            ]
        )
    >>> guard.parse(
    ...   llm_output=...,
    ...   metadata={"query_function": query_function}
    ... )

OR

2. `sources` with an `embed_function` in the metadata. The embed_function should
    take a string or a list of strings as input and return a np array of floats.
The vector should be normalized to unit length.

Example:
    ```py
    def embed_function(text: Union[str, List[str]]) -> np.ndarray:
        return np.array([[0.1, 0.2, 0.3]])

    guard = Guard.from_rail(...)
    guard(
        openai.ChatCompletion.create(...),
        prompt_params={...},
        temperature=0.0,
        metadata={
            "sources": ["This is a source text"],
            "embed_function": embed_function
        },
    )

### call_llm `classfunction`

```
call_llm(
  self,
  prompt: str
) -> <class 'str'>
```

Call the LLM with the given prompt.

Expects a function that takes a string and returns a string.

Args:
    prompt (str): The prompt to send to the LLM.

Returns:
    response (str): String representing the LLM response.

### evaluate_with_llm `classfunction`

```
evaluate_with_llm(
  self,
  text: str,
  query_function: Callable
) -> <class 'bool'>
```

Validate that the LLM-generated text is supported by the provided
contexts.

Args:
    value (Any): The LLM-generated text.
    query_function (Callable): The query function.

Returns:
    self_eval: The self-evaluation boolean

### get_args `classfunction`

```
get_args(
  self
)
```

Get the arguments for the validator.

### get_query_function `classfunction`

```
get_query_function(
  self,
  metadata: Dict[str, Any]
) -> typing.Callable
```

### override_value_on_pass `classbool`

bool(x) -> bool

Returns True when the argument x is true, False otherwise.
The builtins True and False are the only two instances of the class bool.
The class bool is a subclass of the class int, and cannot be subclassed.

### query_vector_collection `classfunction`

```
query_vector_collection(
  text: str,
  k: int,
  sources: List[str],
  embed_function: Callable,
  chunk_strategy: str = 'sentence',
  chunk_size: int = 5,
  chunk_overlap: int = 2,
  distance_metric: str = 'cosine'
) -> typing.List[typing.Tuple[str, float]]
```

### rail_alias `classstr`

str(object='') -> str
str(bytes_or_buffer[, encoding[, errors]]) -> str

Create a new string object from the given object. If encoding or
errors is specified, then the object must expose a data buffer
that will be decoded using the given encoding and error handler.
Otherwise, returns the result of object.__str__() (if defined)
or repr(object).
encoding defaults to sys.getdefaultencoding().
errors defaults to 'strict'.

### required_metadata_keys `classlist`

Built-in mutable sequence.

If no argument is given, the constructor creates a new empty list.
The argument must be an iterable if specified.

### run_in_separate_process `classbool`

bool(x) -> bool

Returns True when the argument x is true, False otherwise.
The builtins True and False are the only two instances of the class bool.
The class bool is a subclass of the class int, and cannot be subclassed.

### set_callable `classfunction`

```
set_callable(
  self,
  llm_callable: Union[str, Callable]
) -> None
```

Set the LLM callable.

Args:
    llm_callable: Either the name of the OpenAI model, or a callable that takes
        a prompt and returns a response.

### to_prompt `classfunction`

```
to_prompt(
  self,
  with_keywords: bool = True
) -> <class 'str'>
```

Convert the validator to a prompt.

E.g. ValidLength(5, 10) -> "length: 5 10" when with_keywords is False.
ValidLength(5, 10) -> "length: min=5 max=10" when with_keywords is True.

Args:
    with_keywords: Whether to include the keyword arguments in the prompt.

Returns:
    A string representation of the validator.

### to_xml_attrib `classfunction`

```
to_xml_attrib(
  self
)
```

Convert the validator to an XML attribute.


## PydanticFieldValidator

Validates a specific field in a Pydantic model with the specified
validator method.

**Key Properties**

| Property                      | Description                       |
| ----------------------------- | --------------------------------- |
| Name for `format` attribute   | `pydantic_field_validator`        |
| Supported data types          | `Any`                             |
| Programmatic fix              | Override with return value from `field_validator`.   |

Parameters: Arguments

    field_validator (Callable): A validator for a specific field in a Pydantic model.

### get_args `classfunction`

```
get_args(
  self
)
```

Get the arguments for the validator.

### override_value_on_pass `classbool`

bool(x) -> bool

Returns True when the argument x is true, False otherwise.
The builtins True and False are the only two instances of the class bool.
The class bool is a subclass of the class int, and cannot be subclassed.

### rail_alias `classstr`

str(object='') -> str
str(bytes_or_buffer[, encoding[, errors]]) -> str

Create a new string object from the given object. If encoding or
errors is specified, then the object must expose a data buffer
that will be decoded using the given encoding and error handler.
Otherwise, returns the result of object.__str__() (if defined)
or repr(object).
encoding defaults to sys.getdefaultencoding().
errors defaults to 'strict'.

### required_metadata_keys `classlist`

Built-in mutable sequence.

If no argument is given, the constructor creates a new empty list.
The argument must be an iterable if specified.

### run_in_separate_process `classbool`

bool(x) -> bool

Returns True when the argument x is true, False otherwise.
The builtins True and False are the only two instances of the class bool.
The class bool is a subclass of the class int, and cannot be subclassed.

### to_prompt `classfunction`

```
to_prompt(
  self,
  with_keywords: bool = True
) -> <class 'str'>
```

Convert the validator to a prompt.

E.g. ValidLength(5, 10) -> "length: 5 10" when with_keywords is False.
ValidLength(5, 10) -> "length: min=5 max=10" when with_keywords is True.

Args:
    with_keywords: Whether to include the keyword arguments in the prompt.

Returns:
    A string representation of the validator.

### to_xml_attrib `classfunction`

```
to_xml_attrib(
  self
)
```

Convert the validator to an XML attribute.


## QARelevanceLLMEval

Validates that an answer is relevant to the question asked by asking the
LLM to self evaluate.

**Key Properties**

| Property                      | Description                         |
| ----------------------------- | ----------------------------------- |
| Name for `format` attribute   | `qa-relevance-llm-eval`             |
| Supported data types          | `string`                            |
| Programmatic fix              | None                                |

Other parameters: Metadata
    question (str): The original question the llm was given to answer.

### _selfeval `classfunction`

```
_selfeval(
  self,
  question: str,
  answer: str
) -> typing.Dict
```

### get_args `classfunction`

```
get_args(
  self
)
```

Get the arguments for the validator.

### override_value_on_pass `classbool`

bool(x) -> bool

Returns True when the argument x is true, False otherwise.
The builtins True and False are the only two instances of the class bool.
The class bool is a subclass of the class int, and cannot be subclassed.

### rail_alias `classstr`

str(object='') -> str
str(bytes_or_buffer[, encoding[, errors]]) -> str

Create a new string object from the given object. If encoding or
errors is specified, then the object must expose a data buffer
that will be decoded using the given encoding and error handler.
Otherwise, returns the result of object.__str__() (if defined)
or repr(object).
encoding defaults to sys.getdefaultencoding().
errors defaults to 'strict'.

### required_metadata_keys `classlist`

Built-in mutable sequence.

If no argument is given, the constructor creates a new empty list.
The argument must be an iterable if specified.

### run_in_separate_process `classbool`

bool(x) -> bool

Returns True when the argument x is true, False otherwise.
The builtins True and False are the only two instances of the class bool.
The class bool is a subclass of the class int, and cannot be subclassed.

### to_prompt `classfunction`

```
to_prompt(
  self,
  with_keywords: bool = True
) -> <class 'str'>
```

Convert the validator to a prompt.

E.g. ValidLength(5, 10) -> "length: 5 10" when with_keywords is False.
ValidLength(5, 10) -> "length: min=5 max=10" when with_keywords is True.

Args:
    with_keywords: Whether to include the keyword arguments in the prompt.

Returns:
    A string representation of the validator.

### to_xml_attrib `classfunction`

```
to_xml_attrib(
  self
)
```

Convert the validator to an XML attribute.


## ReadingTime

Validates that the a string can be read in less than a certain amount of
time.

**Key Properties**

| Property                      | Description                         |
| ----------------------------- | ----------------------------------- |
| Name for `format` attribute   | `reading-time`                      |
| Supported data types          | `string`                            |
| Programmatic fix              | None                                |

Parameters: Arguments

    reading_time: The maximum reading time.

### get_args `classfunction`

```
get_args(
  self
)
```

Get the arguments for the validator.

### override_value_on_pass `classbool`

bool(x) -> bool

Returns True when the argument x is true, False otherwise.
The builtins True and False are the only two instances of the class bool.
The class bool is a subclass of the class int, and cannot be subclassed.

### rail_alias `classstr`

str(object='') -> str
str(bytes_or_buffer[, encoding[, errors]]) -> str

Create a new string object from the given object. If encoding or
errors is specified, then the object must expose a data buffer
that will be decoded using the given encoding and error handler.
Otherwise, returns the result of object.__str__() (if defined)
or repr(object).
encoding defaults to sys.getdefaultencoding().
errors defaults to 'strict'.

### required_metadata_keys `classlist`

Built-in mutable sequence.

If no argument is given, the constructor creates a new empty list.
The argument must be an iterable if specified.

### run_in_separate_process `classbool`

bool(x) -> bool

Returns True when the argument x is true, False otherwise.
The builtins True and False are the only two instances of the class bool.
The class bool is a subclass of the class int, and cannot be subclassed.

### to_prompt `classfunction`

```
to_prompt(
  self,
  with_keywords: bool = True
) -> <class 'str'>
```

Convert the validator to a prompt.

E.g. ValidLength(5, 10) -> "length: 5 10" when with_keywords is False.
ValidLength(5, 10) -> "length: min=5 max=10" when with_keywords is True.

Args:
    with_keywords: Whether to include the keyword arguments in the prompt.

Returns:
    A string representation of the validator.

### to_xml_attrib `classfunction`

```
to_xml_attrib(
  self
)
```

Convert the validator to an XML attribute.


## RegexMatch

Validates that a value matches a regular expression.

**Key Properties**

| Property                      | Description                       |
| ----------------------------- | --------------------------------- |
| Name for `format` attribute   | `regex_match`                     |
| Supported data types          | `string`                          |
| Programmatic fix              | Generate a string that matches the regular expression |

Parameters: Arguments
    regex: Str regex pattern
    match_type: Str in {"search", "fullmatch"} for a regex search or full-match option

### get_args `classfunction`

```
get_args(
  self
)
```

Get the arguments for the validator.

### override_value_on_pass `classbool`

bool(x) -> bool

Returns True when the argument x is true, False otherwise.
The builtins True and False are the only two instances of the class bool.
The class bool is a subclass of the class int, and cannot be subclassed.

### rail_alias `classstr`

str(object='') -> str
str(bytes_or_buffer[, encoding[, errors]]) -> str

Create a new string object from the given object. If encoding or
errors is specified, then the object must expose a data buffer
that will be decoded using the given encoding and error handler.
Otherwise, returns the result of object.__str__() (if defined)
or repr(object).
encoding defaults to sys.getdefaultencoding().
errors defaults to 'strict'.

### required_metadata_keys `classlist`

Built-in mutable sequence.

If no argument is given, the constructor creates a new empty list.
The argument must be an iterable if specified.

### run_in_separate_process `classbool`

bool(x) -> bool

Returns True when the argument x is true, False otherwise.
The builtins True and False are the only two instances of the class bool.
The class bool is a subclass of the class int, and cannot be subclassed.

### to_prompt `classfunction`

```
to_prompt(
  self,
  with_keywords: bool = True
) -> <class 'str'>
```

Convert the validator to a prompt.

E.g. ValidLength(5, 10) -> "length: 5 10" when with_keywords is False.
ValidLength(5, 10) -> "length: min=5 max=10" when with_keywords is True.

Args:
    with_keywords: Whether to include the keyword arguments in the prompt.

Returns:
    A string representation of the validator.

### to_xml_attrib `classfunction`

```
to_xml_attrib(
  self
)
```

Convert the validator to an XML attribute.


## RemoveRedundantSentences

Removes redundant sentences from a string.

This validator removes sentences from a string that are similar to
other sentences in the string. This is useful for removing
repetitive sentences from a string.

**Key Properties**

| Property                      | Description                         |
| ----------------------------- | ----------------------------------- |
| Name for `format` attribute   | `remove-redundant-sentences`        |
| Supported data types          | `string`                            |
| Programmatic fix              | Remove any redundant sentences.     |

Parameters: Arguments

    threshold: The minimum fuzz ratio to be considered redundant.  Defaults to 70.

### get_args `classfunction`

```
get_args(
  self
)
```

Get the arguments for the validator.

### override_value_on_pass `classbool`

bool(x) -> bool

Returns True when the argument x is true, False otherwise.
The builtins True and False are the only two instances of the class bool.
The class bool is a subclass of the class int, and cannot be subclassed.

### rail_alias `classstr`

str(object='') -> str
str(bytes_or_buffer[, encoding[, errors]]) -> str

Create a new string object from the given object. If encoding or
errors is specified, then the object must expose a data buffer
that will be decoded using the given encoding and error handler.
Otherwise, returns the result of object.__str__() (if defined)
or repr(object).
encoding defaults to sys.getdefaultencoding().
errors defaults to 'strict'.

### required_metadata_keys `classlist`

Built-in mutable sequence.

If no argument is given, the constructor creates a new empty list.
The argument must be an iterable if specified.

### run_in_separate_process `classbool`

bool(x) -> bool

Returns True when the argument x is true, False otherwise.
The builtins True and False are the only two instances of the class bool.
The class bool is a subclass of the class int, and cannot be subclassed.

### to_prompt `classfunction`

```
to_prompt(
  self,
  with_keywords: bool = True
) -> <class 'str'>
```

Convert the validator to a prompt.

E.g. ValidLength(5, 10) -> "length: 5 10" when with_keywords is False.
ValidLength(5, 10) -> "length: min=5 max=10" when with_keywords is True.

Args:
    with_keywords: Whether to include the keyword arguments in the prompt.

Returns:
    A string representation of the validator.

### to_xml_attrib `classfunction`

```
to_xml_attrib(
  self
)
```

Convert the validator to an XML attribute.


## SaliencyCheck

Checks that the summary covers the list of topics present in the
document.

**Key Properties**

| Property                      | Description                         |
| ----------------------------- | ----------------------------------- |
| Name for `format` attribute   | `saliency-check`                    |
| Supported data types          | `string`                            |
| Programmatic fix              | None                                |

Parameters: Arguments

    docs_dir: Path to the directory containing the documents.
    threshold: Threshold for overlap between topics in document and summary. Defaults to 0.25

### _get_topics `classfunction`

```
_get_topics(
  self,
  text: str,
  topics: Optional[List[str]] = None
) -> typing.List[str]
```

Extract topics from a string.

### _topics `classproperty`

Return a list of topics that can be used in the validator.

### get_args `classfunction`

```
get_args(
  self
)
```

Get the arguments for the validator.

### override_value_on_pass `classbool`

bool(x) -> bool

Returns True when the argument x is true, False otherwise.
The builtins True and False are the only two instances of the class bool.
The class bool is a subclass of the class int, and cannot be subclassed.

### rail_alias `classstr`

str(object='') -> str
str(bytes_or_buffer[, encoding[, errors]]) -> str

Create a new string object from the given object. If encoding or
errors is specified, then the object must expose a data buffer
that will be decoded using the given encoding and error handler.
Otherwise, returns the result of object.__str__() (if defined)
or repr(object).
encoding defaults to sys.getdefaultencoding().
errors defaults to 'strict'.

### required_metadata_keys `classlist`

Built-in mutable sequence.

If no argument is given, the constructor creates a new empty list.
The argument must be an iterable if specified.

### run_in_separate_process `classbool`

bool(x) -> bool

Returns True when the argument x is true, False otherwise.
The builtins True and False are the only two instances of the class bool.
The class bool is a subclass of the class int, and cannot be subclassed.

### to_prompt `classfunction`

```
to_prompt(
  self,
  with_keywords: bool = True
) -> <class 'str'>
```

Convert the validator to a prompt.

E.g. ValidLength(5, 10) -> "length: 5 10" when with_keywords is False.
ValidLength(5, 10) -> "length: min=5 max=10" when with_keywords is True.

Args:
    with_keywords: Whether to include the keyword arguments in the prompt.

Returns:
    A string representation of the validator.

### to_xml_attrib `classfunction`

```
to_xml_attrib(
  self
)
```

Convert the validator to an XML attribute.


## SimilarToDocument

Validates that a value is similar to the document.

This validator checks if the value is similar to the document by checking
the cosine similarity between the value and the document, using an
embedding.

**Key Properties**

| Property                      | Description                       |
| ----------------------------- | --------------------------------- |
| Name for `format` attribute   | `similar-to-document`             |
| Supported data types          | `string`                             |
| Programmatic fix              | None                              |

Parameters: Arguments
    document: The document to use for the similarity check.
    threshold: The minimum cosine similarity to be considered similar.  Defaults to 0.7.
    model: The embedding model to use.  Defaults to text-embedding-ada-002.

### cosine_similarity `classfunction`

```
cosine_similarity(
  a: 'np.ndarray',
  b: 'np.ndarray'
) -> <class 'float'>
```

Calculate the cosine similarity between two vectors.

Args:
    a: The first vector.
    b: The second vector.

Returns:
    float: The cosine similarity between the two vectors.

### get_args `classfunction`

```
get_args(
  self
)
```

Get the arguments for the validator.

### override_value_on_pass `classbool`

bool(x) -> bool

Returns True when the argument x is true, False otherwise.
The builtins True and False are the only two instances of the class bool.
The class bool is a subclass of the class int, and cannot be subclassed.

### rail_alias `classstr`

str(object='') -> str
str(bytes_or_buffer[, encoding[, errors]]) -> str

Create a new string object from the given object. If encoding or
errors is specified, then the object must expose a data buffer
that will be decoded using the given encoding and error handler.
Otherwise, returns the result of object.__str__() (if defined)
or repr(object).
encoding defaults to sys.getdefaultencoding().
errors defaults to 'strict'.

### required_metadata_keys `classlist`

Built-in mutable sequence.

If no argument is given, the constructor creates a new empty list.
The argument must be an iterable if specified.

### run_in_separate_process `classbool`

bool(x) -> bool

Returns True when the argument x is true, False otherwise.
The builtins True and False are the only two instances of the class bool.
The class bool is a subclass of the class int, and cannot be subclassed.

### to_prompt `classfunction`

```
to_prompt(
  self,
  with_keywords: bool = True
) -> <class 'str'>
```

Convert the validator to a prompt.

E.g. ValidLength(5, 10) -> "length: 5 10" when with_keywords is False.
ValidLength(5, 10) -> "length: min=5 max=10" when with_keywords is True.

Args:
    with_keywords: Whether to include the keyword arguments in the prompt.

Returns:
    A string representation of the validator.

### to_xml_attrib `classfunction`

```
to_xml_attrib(
  self
)
```

Convert the validator to an XML attribute.


## SimilarToList

Validates that a value is similar to a list of previously known values.

**Key Properties**

| Property                      | Description                       |
| ----------------------------- | --------------------------------- |
| Name for `format` attribute   | `similar-to-list`                 |
| Supported data types          | `string`                          |
| Programmatic fix              | None                              |

Parameters: Arguments
    standard_deviations (int): The number of standard deviations from the mean to check.
    threshold (float): The threshold for the average semantic similarity for strings.

For integer values, this validator checks whether the value lies
within 'k' standard deviations of the mean of the previous values.
(Assumes that the previous values are normally distributed.) For
string values, this validator checks whether the average semantic
similarity between the generated value and the previous values is
less than a threshold.

### get_args `classfunction`

```
get_args(
  self
)
```

Get the arguments for the validator.

### get_semantic_similarity `classfunction`

```
get_semantic_similarity(
  self,
  text1: str,
  text2: str,
  embed_function: Callable
) -> <class 'float'>
```

Get the semantic similarity between two strings.

Args:
    text1 (str): The first string.
    text2 (str): The second string.
    embed_function (Callable): The embedding function.
Returns:
    similarity (float): The semantic similarity between the two strings.

### override_value_on_pass `classbool`

bool(x) -> bool

Returns True when the argument x is true, False otherwise.
The builtins True and False are the only two instances of the class bool.
The class bool is a subclass of the class int, and cannot be subclassed.

### rail_alias `classstr`

str(object='') -> str
str(bytes_or_buffer[, encoding[, errors]]) -> str

Create a new string object from the given object. If encoding or
errors is specified, then the object must expose a data buffer
that will be decoded using the given encoding and error handler.
Otherwise, returns the result of object.__str__() (if defined)
or repr(object).
encoding defaults to sys.getdefaultencoding().
errors defaults to 'strict'.

### required_metadata_keys `classlist`

Built-in mutable sequence.

If no argument is given, the constructor creates a new empty list.
The argument must be an iterable if specified.

### run_in_separate_process `classbool`

bool(x) -> bool

Returns True when the argument x is true, False otherwise.
The builtins True and False are the only two instances of the class bool.
The class bool is a subclass of the class int, and cannot be subclassed.

### to_prompt `classfunction`

```
to_prompt(
  self,
  with_keywords: bool = True
) -> <class 'str'>
```

Convert the validator to a prompt.

E.g. ValidLength(5, 10) -> "length: 5 10" when with_keywords is False.
ValidLength(5, 10) -> "length: min=5 max=10" when with_keywords is True.

Args:
    with_keywords: Whether to include the keyword arguments in the prompt.

Returns:
    A string representation of the validator.

### to_xml_attrib `classfunction`

```
to_xml_attrib(
  self
)
```

Convert the validator to an XML attribute.


## SqlColumnPresence

Validates that all columns in the SQL query are present in the schema.

**Key Properties**

| Property                      | Description                       |
| ----------------------------- | --------------------------------- |
| Name for `format` attribute   | `sql-column-presence`             |
| Supported data types          | `string`                          |
| Programmatic fix              | None                              |

Parameters: Arguments
    cols: The list of valid columns.

### get_args `classfunction`

```
get_args(
  self
)
```

Get the arguments for the validator.

### override_value_on_pass `classbool`

bool(x) -> bool

Returns True when the argument x is true, False otherwise.
The builtins True and False are the only two instances of the class bool.
The class bool is a subclass of the class int, and cannot be subclassed.

### rail_alias `classstr`

str(object='') -> str
str(bytes_or_buffer[, encoding[, errors]]) -> str

Create a new string object from the given object. If encoding or
errors is specified, then the object must expose a data buffer
that will be decoded using the given encoding and error handler.
Otherwise, returns the result of object.__str__() (if defined)
or repr(object).
encoding defaults to sys.getdefaultencoding().
errors defaults to 'strict'.

### required_metadata_keys `classlist`

Built-in mutable sequence.

If no argument is given, the constructor creates a new empty list.
The argument must be an iterable if specified.

### run_in_separate_process `classbool`

bool(x) -> bool

Returns True when the argument x is true, False otherwise.
The builtins True and False are the only two instances of the class bool.
The class bool is a subclass of the class int, and cannot be subclassed.

### to_prompt `classfunction`

```
to_prompt(
  self,
  with_keywords: bool = True
) -> <class 'str'>
```

Convert the validator to a prompt.

E.g. ValidLength(5, 10) -> "length: 5 10" when with_keywords is False.
ValidLength(5, 10) -> "length: min=5 max=10" when with_keywords is True.

Args:
    with_keywords: Whether to include the keyword arguments in the prompt.

Returns:
    A string representation of the validator.

### to_xml_attrib `classfunction`

```
to_xml_attrib(
  self
)
```

Convert the validator to an XML attribute.


## ToxicLanguage

Validates that the generated text is toxic.

**Key Properties**
| Property                      | Description                       |
| ----------------------------- | --------------------------------- |
| Name for `format` attribute   | `toxic-language`                  |
| Supported data types          | `string`                          |
| Programmatic fix              | None                              |

Parameters: Arguments
    threshold: The confidence threshold (model inference) for toxicity.
        Defaults to 0.5.
    validation_method: Whether to validate at the sentence level or
        over the full text. Must be one of `sentence` or `full`.
        Defaults to `sentence`

This validator uses the pre-trained multi-label model from HuggingFace -
`unitary/unbiased-toxic-roberta` to check whether the generated text is toxic.
If the model predicts any label of: `toxicity`, `severe_toxicity`,
`obscene`, `threat`, `insult`, `identity_attack`, or `sexual_explicit` with
confidence higher than the specified threshold, the validator fails and returns
the generated text with the toxic sentences / entire text removed. Else the
validator returns the generated text as it is.

If validation_method is `sentence`, the validator will remove the sentences
that are predicted to be toxic and return the remaining sentences. If
validation_method is `full`, the validator will remove the entire text if
the prediction is deemed toxic and return an empty string.

In our experiments, a threshold of 0.5 worked best, hence set as default here.
However, you can try different values of threshold to see what works best for
your use case.
Link for experiments: https://wandb.ai/ml-guardrails/toxic-language-experiments

### get_args `classfunction`

```
get_args(
  self
)
```

Get the arguments for the validator.

### get_toxicity `classfunction`

```
get_toxicity(
  self,
  value: str
) -> typing.List[str]
```

Check whether the generated text is toxic.

Returns the labels predicted by the model with
confidence higher than the threshold.

Args:
    value (str): The generated text.

Returns:
    pred_labels (bool): Labels predicted by the model
    with confidence higher than the threshold.

### override_value_on_pass `classbool`

bool(x) -> bool

Returns True when the argument x is true, False otherwise.
The builtins True and False are the only two instances of the class bool.
The class bool is a subclass of the class int, and cannot be subclassed.

### rail_alias `classstr`

str(object='') -> str
str(bytes_or_buffer[, encoding[, errors]]) -> str

Create a new string object from the given object. If encoding or
errors is specified, then the object must expose a data buffer
that will be decoded using the given encoding and error handler.
Otherwise, returns the result of object.__str__() (if defined)
or repr(object).
encoding defaults to sys.getdefaultencoding().
errors defaults to 'strict'.

### required_metadata_keys `classlist`

Built-in mutable sequence.

If no argument is given, the constructor creates a new empty list.
The argument must be an iterable if specified.

### run_in_separate_process `classbool`

bool(x) -> bool

Returns True when the argument x is true, False otherwise.
The builtins True and False are the only two instances of the class bool.
The class bool is a subclass of the class int, and cannot be subclassed.

### to_prompt `classfunction`

```
to_prompt(
  self,
  with_keywords: bool = True
) -> <class 'str'>
```

Convert the validator to a prompt.

E.g. ValidLength(5, 10) -> "length: 5 10" when with_keywords is False.
ValidLength(5, 10) -> "length: min=5 max=10" when with_keywords is True.

Args:
    with_keywords: Whether to include the keyword arguments in the prompt.

Returns:
    A string representation of the validator.

### to_xml_attrib `classfunction`

```
to_xml_attrib(
  self
)
```

Convert the validator to an XML attribute.


## TwoWords

Validates that a value is two words.

**Key Properties**

| Property                      | Description                       |
| ----------------------------- | --------------------------------- |
| Name for `format` attribute   | `two-words`                       |
| Supported data types          | `string`                          |
| Programmatic fix              | Pick the first two words.         |

### _get_fix_value `classfunction`

```
_get_fix_value(
  self,
  value: str
) -> <class 'str'>
```

### get_args `classfunction`

```
get_args(
  self
)
```

Get the arguments for the validator.

### override_value_on_pass `classbool`

bool(x) -> bool

Returns True when the argument x is true, False otherwise.
The builtins True and False are the only two instances of the class bool.
The class bool is a subclass of the class int, and cannot be subclassed.

### rail_alias `classstr`

str(object='') -> str
str(bytes_or_buffer[, encoding[, errors]]) -> str

Create a new string object from the given object. If encoding or
errors is specified, then the object must expose a data buffer
that will be decoded using the given encoding and error handler.
Otherwise, returns the result of object.__str__() (if defined)
or repr(object).
encoding defaults to sys.getdefaultencoding().
errors defaults to 'strict'.

### required_metadata_keys `classlist`

Built-in mutable sequence.

If no argument is given, the constructor creates a new empty list.
The argument must be an iterable if specified.

### run_in_separate_process `classbool`

bool(x) -> bool

Returns True when the argument x is true, False otherwise.
The builtins True and False are the only two instances of the class bool.
The class bool is a subclass of the class int, and cannot be subclassed.

### to_prompt `classfunction`

```
to_prompt(
  self,
  with_keywords: bool = True
) -> <class 'str'>
```

Convert the validator to a prompt.

E.g. ValidLength(5, 10) -> "length: 5 10" when with_keywords is False.
ValidLength(5, 10) -> "length: min=5 max=10" when with_keywords is True.

Args:
    with_keywords: Whether to include the keyword arguments in the prompt.

Returns:
    A string representation of the validator.

### to_xml_attrib `classfunction`

```
to_xml_attrib(
  self
)
```

Convert the validator to an XML attribute.


## UpperCase

Validates that a value is upper case.

**Key Properties**

| Property                      | Description                       |
| ----------------------------- | --------------------------------- |
| Name for `format` attribute   | `upper-case`                      |
| Supported data types          | `string`                          |
| Programmatic fix              | Convert to upper case.            |

### get_args `classfunction`

```
get_args(
  self
)
```

Get the arguments for the validator.

### override_value_on_pass `classbool`

bool(x) -> bool

Returns True when the argument x is true, False otherwise.
The builtins True and False are the only two instances of the class bool.
The class bool is a subclass of the class int, and cannot be subclassed.

### rail_alias `classstr`

str(object='') -> str
str(bytes_or_buffer[, encoding[, errors]]) -> str

Create a new string object from the given object. If encoding or
errors is specified, then the object must expose a data buffer
that will be decoded using the given encoding and error handler.
Otherwise, returns the result of object.__str__() (if defined)
or repr(object).
encoding defaults to sys.getdefaultencoding().
errors defaults to 'strict'.

### required_metadata_keys `classlist`

Built-in mutable sequence.

If no argument is given, the constructor creates a new empty list.
The argument must be an iterable if specified.

### run_in_separate_process `classbool`

bool(x) -> bool

Returns True when the argument x is true, False otherwise.
The builtins True and False are the only two instances of the class bool.
The class bool is a subclass of the class int, and cannot be subclassed.

### to_prompt `classfunction`

```
to_prompt(
  self,
  with_keywords: bool = True
) -> <class 'str'>
```

Convert the validator to a prompt.

E.g. ValidLength(5, 10) -> "length: 5 10" when with_keywords is False.
ValidLength(5, 10) -> "length: min=5 max=10" when with_keywords is True.

Args:
    with_keywords: Whether to include the keyword arguments in the prompt.

Returns:
    A string representation of the validator.

### to_xml_attrib `classfunction`

```
to_xml_attrib(
  self
)
```

Convert the validator to an XML attribute.


## ValidChoices

Validates that a value is within the acceptable choices.

**Key Properties**

| Property                      | Description                       |
| ----------------------------- | --------------------------------- |
| Name for `format` attribute   | `valid-choices`                   |
| Supported data types          | `all`                             |
| Programmatic fix              | None                              |

Parameters: Arguments
    choices: The list of valid choices.

### get_args `classfunction`

```
get_args(
  self
)
```

Get the arguments for the validator.

### override_value_on_pass `classbool`

bool(x) -> bool

Returns True when the argument x is true, False otherwise.
The builtins True and False are the only two instances of the class bool.
The class bool is a subclass of the class int, and cannot be subclassed.

### rail_alias `classstr`

str(object='') -> str
str(bytes_or_buffer[, encoding[, errors]]) -> str

Create a new string object from the given object. If encoding or
errors is specified, then the object must expose a data buffer
that will be decoded using the given encoding and error handler.
Otherwise, returns the result of object.__str__() (if defined)
or repr(object).
encoding defaults to sys.getdefaultencoding().
errors defaults to 'strict'.

### required_metadata_keys `classlist`

Built-in mutable sequence.

If no argument is given, the constructor creates a new empty list.
The argument must be an iterable if specified.

### run_in_separate_process `classbool`

bool(x) -> bool

Returns True when the argument x is true, False otherwise.
The builtins True and False are the only two instances of the class bool.
The class bool is a subclass of the class int, and cannot be subclassed.

### to_prompt `classfunction`

```
to_prompt(
  self,
  with_keywords: bool = True
) -> <class 'str'>
```

Convert the validator to a prompt.

E.g. ValidLength(5, 10) -> "length: 5 10" when with_keywords is False.
ValidLength(5, 10) -> "length: min=5 max=10" when with_keywords is True.

Args:
    with_keywords: Whether to include the keyword arguments in the prompt.

Returns:
    A string representation of the validator.

### to_xml_attrib `classfunction`

```
to_xml_attrib(
  self
)
```

Convert the validator to an XML attribute.


## ValidLength

Validates that the length of value is within the expected range.

**Key Properties**

| Property                      | Description                       |
| ----------------------------- | --------------------------------- |
| Name for `format` attribute   | `length`                          |
| Supported data types          | `string`, `list`, `object`        |
| Programmatic fix              | If shorter than the minimum, pad with empty last elements. If longer than the maximum, truncate. |

Parameters: Arguments
    min: The inclusive minimum length.
    max: The inclusive maximum length.

### get_args `classfunction`

```
get_args(
  self
)
```

Get the arguments for the validator.

### override_value_on_pass `classbool`

bool(x) -> bool

Returns True when the argument x is true, False otherwise.
The builtins True and False are the only two instances of the class bool.
The class bool is a subclass of the class int, and cannot be subclassed.

### rail_alias `classstr`

str(object='') -> str
str(bytes_or_buffer[, encoding[, errors]]) -> str

Create a new string object from the given object. If encoding or
errors is specified, then the object must expose a data buffer
that will be decoded using the given encoding and error handler.
Otherwise, returns the result of object.__str__() (if defined)
or repr(object).
encoding defaults to sys.getdefaultencoding().
errors defaults to 'strict'.

### required_metadata_keys `classlist`

Built-in mutable sequence.

If no argument is given, the constructor creates a new empty list.
The argument must be an iterable if specified.

### run_in_separate_process `classbool`

bool(x) -> bool

Returns True when the argument x is true, False otherwise.
The builtins True and False are the only two instances of the class bool.
The class bool is a subclass of the class int, and cannot be subclassed.

### to_prompt `classfunction`

```
to_prompt(
  self,
  with_keywords: bool = True
) -> <class 'str'>
```

Convert the validator to a prompt.

E.g. ValidLength(5, 10) -> "length: 5 10" when with_keywords is False.
ValidLength(5, 10) -> "length: min=5 max=10" when with_keywords is True.

Args:
    with_keywords: Whether to include the keyword arguments in the prompt.

Returns:
    A string representation of the validator.

### to_xml_attrib `classfunction`

```
to_xml_attrib(
  self
)
```

Convert the validator to an XML attribute.


## ValidRange

Validates that a value is within a range.

**Key Properties**

| Property                      | Description                       |
| ----------------------------- | --------------------------------- |
| Name for `format` attribute   | `valid-range`                     |
| Supported data types          | `integer`, `float`, `percentage`  |
| Programmatic fix              | Closest value within the range.   |

Parameters: Arguments
    min: The inclusive minimum value of the range.
    max: The inclusive maximum value of the range.

### get_args `classfunction`

```
get_args(
  self
)
```

Get the arguments for the validator.

### override_value_on_pass `classbool`

bool(x) -> bool

Returns True when the argument x is true, False otherwise.
The builtins True and False are the only two instances of the class bool.
The class bool is a subclass of the class int, and cannot be subclassed.

### rail_alias `classstr`

str(object='') -> str
str(bytes_or_buffer[, encoding[, errors]]) -> str

Create a new string object from the given object. If encoding or
errors is specified, then the object must expose a data buffer
that will be decoded using the given encoding and error handler.
Otherwise, returns the result of object.__str__() (if defined)
or repr(object).
encoding defaults to sys.getdefaultencoding().
errors defaults to 'strict'.

### required_metadata_keys `classlist`

Built-in mutable sequence.

If no argument is given, the constructor creates a new empty list.
The argument must be an iterable if specified.

### run_in_separate_process `classbool`

bool(x) -> bool

Returns True when the argument x is true, False otherwise.
The builtins True and False are the only two instances of the class bool.
The class bool is a subclass of the class int, and cannot be subclassed.

### to_prompt `classfunction`

```
to_prompt(
  self,
  with_keywords: bool = True
) -> <class 'str'>
```

Convert the validator to a prompt.

E.g. ValidLength(5, 10) -> "length: 5 10" when with_keywords is False.
ValidLength(5, 10) -> "length: min=5 max=10" when with_keywords is True.

Args:
    with_keywords: Whether to include the keyword arguments in the prompt.

Returns:
    A string representation of the validator.

### to_xml_attrib `classfunction`

```
to_xml_attrib(
  self
)
```

Convert the validator to an XML attribute.


## ValidURL

Validates that a value is a valid URL.

**Key Properties**

| Property                      | Description                       |
| ----------------------------- | --------------------------------- |
| Name for `format` attribute   | `valid-url`                       |
| Supported data types          | `string`                          |
| Programmatic fix              | None                              |

### get_args `classfunction`

```
get_args(
  self
)
```

Get the arguments for the validator.

### override_value_on_pass `classbool`

bool(x) -> bool

Returns True when the argument x is true, False otherwise.
The builtins True and False are the only two instances of the class bool.
The class bool is a subclass of the class int, and cannot be subclassed.

### rail_alias `classstr`

str(object='') -> str
str(bytes_or_buffer[, encoding[, errors]]) -> str

Create a new string object from the given object. If encoding or
errors is specified, then the object must expose a data buffer
that will be decoded using the given encoding and error handler.
Otherwise, returns the result of object.__str__() (if defined)
or repr(object).
encoding defaults to sys.getdefaultencoding().
errors defaults to 'strict'.

### required_metadata_keys `classlist`

Built-in mutable sequence.

If no argument is given, the constructor creates a new empty list.
The argument must be an iterable if specified.

### run_in_separate_process `classbool`

bool(x) -> bool

Returns True when the argument x is true, False otherwise.
The builtins True and False are the only two instances of the class bool.
The class bool is a subclass of the class int, and cannot be subclassed.

### to_prompt `classfunction`

```
to_prompt(
  self,
  with_keywords: bool = True
) -> <class 'str'>
```

Convert the validator to a prompt.

E.g. ValidLength(5, 10) -> "length: 5 10" when with_keywords is False.
ValidLength(5, 10) -> "length: min=5 max=10" when with_keywords is True.

Args:
    with_keywords: Whether to include the keyword arguments in the prompt.

Returns:
    A string representation of the validator.

### to_xml_attrib `classfunction`

```
to_xml_attrib(
  self
)
```

Convert the validator to an XML attribute.


### pipeline `classfunction`

```
pipeline(
  task: str = None,
  model: Union[str, ForwardRef('PreTrainedModel'), ForwardRef('TFPreTrainedModel'), NoneType] = None,
  config: Union[str, transformers.configuration_utils.PretrainedConfig, NoneType] = None,
  tokenizer: Union[str, transformers.tokenization_utils.PreTrainedTokenizer, ForwardRef('PreTrainedTokenizerFast'), NoneType] = None,
  feature_extractor: Union[str, ForwardRef('SequenceFeatureExtractor'), NoneType] = None,
  image_processor: Union[str, transformers.image_processing_utils.BaseImageProcessor, NoneType] = None,
  framework: Optional[str] = None,
  revision: Optional[str] = None,
  use_fast: bool = True,
  token: Union[str, bool, NoneType] = None,
  device: Union[int, str, ForwardRef('torch.device'), NoneType] = None,
  device_map=None,
  torch_dtype=None,
  trust_remote_code: Optional[bool] = None,
  model_kwargs: Dict[str, Any] = None,
  pipeline_class: Optional[Any] = None,
  **kwargs
) -> <class 'transformers.pipelines.base.Pipeline'>
```

Utility factory method to build a [`Pipeline`].

Pipelines are made of:

    - A [tokenizer](tokenizer) in charge of mapping raw textual input to token.
    - A [model](model) to make predictions from the inputs.
    - Some (optional) post processing for enhancing model's output.

Args:
    task (`str`):
        The task defining which pipeline will be returned. Currently accepted tasks are:

        - `"audio-classification"`: will return a [`AudioClassificationPipeline`].
        - `"automatic-speech-recognition"`: will return a [`AutomaticSpeechRecognitionPipeline`].
        - `"conversational"`: will return a [`ConversationalPipeline`].
        - `"depth-estimation"`: will return a [`DepthEstimationPipeline`].
        - `"document-question-answering"`: will return a [`DocumentQuestionAnsweringPipeline`].
        - `"feature-extraction"`: will return a [`FeatureExtractionPipeline`].
        - `"fill-mask"`: will return a [`FillMaskPipeline`]:.
        - `"image-classification"`: will return a [`ImageClassificationPipeline`].
        - `"image-segmentation"`: will return a [`ImageSegmentationPipeline`].
        - `"image-to-image"`: will return a [`ImageToImagePipeline`].
        - `"image-to-text"`: will return a [`ImageToTextPipeline`].
        - `"mask-generation"`: will return a [`MaskGenerationPipeline`].
        - `"object-detection"`: will return a [`ObjectDetectionPipeline`].
        - `"question-answering"`: will return a [`QuestionAnsweringPipeline`].
        - `"summarization"`: will return a [`SummarizationPipeline`].
        - `"table-question-answering"`: will return a [`TableQuestionAnsweringPipeline`].
        - `"text2text-generation"`: will return a [`Text2TextGenerationPipeline`].
        - `"text-classification"` (alias `"sentiment-analysis"` available): will return a
          [`TextClassificationPipeline`].
        - `"text-generation"`: will return a [`TextGenerationPipeline`]:.
        - `"text-to-audio"` (alias `"text-to-speech"` available): will return a [`TextToAudioPipeline`]:.
        - `"token-classification"` (alias `"ner"` available): will return a [`TokenClassificationPipeline`].
        - `"translation"`: will return a [`TranslationPipeline`].
        - `"translation_xx_to_yy"`: will return a [`TranslationPipeline`].
        - `"video-classification"`: will return a [`VideoClassificationPipeline`].
        - `"visual-question-answering"`: will return a [`VisualQuestionAnsweringPipeline`].
        - `"zero-shot-classification"`: will return a [`ZeroShotClassificationPipeline`].
        - `"zero-shot-image-classification"`: will return a [`ZeroShotImageClassificationPipeline`].
        - `"zero-shot-audio-classification"`: will return a [`ZeroShotAudioClassificationPipeline`].
        - `"zero-shot-object-detection"`: will return a [`ZeroShotObjectDetectionPipeline`].

    model (`str` or [`PreTrainedModel`] or [`TFPreTrainedModel`], *optional*):
        The model that will be used by the pipeline to make predictions. This can be a model identifier or an
        actual instance of a pretrained model inheriting from [`PreTrainedModel`] (for PyTorch) or
        [`TFPreTrainedModel`] (for TensorFlow).

        If not provided, the default for the `task` will be loaded.
    config (`str` or [`PretrainedConfig`], *optional*):
        The configuration that will be used by the pipeline to instantiate the model. This can be a model
        identifier or an actual pretrained model configuration inheriting from [`PretrainedConfig`].

        If not provided, the default configuration file for the requested model will be used. That means that if
        `model` is given, its default configuration will be used. However, if `model` is not supplied, this
        `task`'s default model's config is used instead.
    tokenizer (`str` or [`PreTrainedTokenizer`], *optional*):
        The tokenizer that will be used by the pipeline to encode data for the model. This can be a model
        identifier or an actual pretrained tokenizer inheriting from [`PreTrainedTokenizer`].

        If not provided, the default tokenizer for the given `model` will be loaded (if it is a string). If `model`
        is not specified or not a string, then the default tokenizer for `config` is loaded (if it is a string).
        However, if `config` is also not given or not a string, then the default tokenizer for the given `task`
        will be loaded.
    feature_extractor (`str` or [`PreTrainedFeatureExtractor`], *optional*):
        The feature extractor that will be used by the pipeline to encode data for the model. This can be a model
        identifier or an actual pretrained feature extractor inheriting from [`PreTrainedFeatureExtractor`].

        Feature extractors are used for non-NLP models, such as Speech or Vision models as well as multi-modal
        models. Multi-modal models will also require a tokenizer to be passed.

        If not provided, the default feature extractor for the given `model` will be loaded (if it is a string). If
        `model` is not specified or not a string, then the default feature extractor for `config` is loaded (if it
        is a string). However, if `config` is also not given or not a string, then the default feature extractor
        for the given `task` will be loaded.
    framework (`str`, *optional*):
        The framework to use, either `"pt"` for PyTorch or `"tf"` for TensorFlow. The specified framework must be
        installed.

        If no framework is specified, will default to the one currently installed. If no framework is specified and
        both frameworks are installed, will default to the framework of the `model`, or to PyTorch if no model is
        provided.
    revision (`str`, *optional*, defaults to `"main"`):
        When passing a task name or a string model identifier: The specific model version to use. It can be a
        branch name, a tag name, or a commit id, since we use a git-based system for storing models and other
        artifacts on huggingface.co, so `revision` can be any identifier allowed by git.
    use_fast (`bool`, *optional*, defaults to `True`):
        Whether or not to use a Fast tokenizer if possible (a [`PreTrainedTokenizerFast`]).
    use_auth_token (`str` or *bool*, *optional*):
        The token to use as HTTP bearer authorization for remote files. If `True`, will use the token generated
        when running `huggingface-cli login` (stored in `~/.huggingface`).
    device (`int` or `str` or `torch.device`):
        Defines the device (*e.g.*, `"cpu"`, `"cuda:1"`, `"mps"`, or a GPU ordinal rank like `1`) on which this
        pipeline will be allocated.
    device_map (`str` or `Dict[str, Union[int, str, torch.device]`, *optional*):
        Sent directly as `model_kwargs` (just a simpler shortcut). When `accelerate` library is present, set
        `device_map="auto"` to compute the most optimized `device_map` automatically (see
        [here](https://huggingface.co/docs/accelerate/main/en/package_reference/big_modeling#accelerate.cpu_offload)
        for more information).

        <Tip warning={true}>

        Do not use `device_map` AND `device` at the same time as they will conflict

        </Tip>

    torch_dtype (`str` or `torch.dtype`, *optional*):
        Sent directly as `model_kwargs` (just a simpler shortcut) to use the available precision for this model
        (`torch.float16`, `torch.bfloat16`, ... or `"auto"`).
    trust_remote_code (`bool`, *optional*, defaults to `False`):
        Whether or not to allow for custom code defined on the Hub in their own modeling, configuration,
        tokenization or even pipeline files. This option should only be set to `True` for repositories you trust
        and in which you have read the code, as it will execute code present on the Hub on your local machine.
    model_kwargs (`Dict[str, Any]`, *optional*):
        Additional dictionary of keyword arguments passed along to the model's `from_pretrained(...,
        **model_kwargs)` function.
    kwargs (`Dict[str, Any]`, *optional*):
        Additional keyword arguments passed along to the specific pipeline init (see the documentation for the
        corresponding pipeline class for possible values).

Returns:
    [`Pipeline`]: A suitable pipeline for the task.

Examples:

```python
>>> from transformers import pipeline, AutoModelForTokenClassification, AutoTokenizer

>>> # Sentiment analysis pipeline
>>> analyzer = pipeline("sentiment-analysis")

>>> # Question answering pipeline, specifying the checkpoint identifier
>>> oracle = pipeline(
...     "question-answering", model="distilbert-base-cased-distilled-squad", tokenizer="bert-base-cased"
... )

>>> # Named entity recognition pipeline, passing in a specific model and tokenizer
>>> model = AutoModelForTokenClassification.from_pretrained("dbmdz/bert-large-cased-finetuned-conll03-english")
>>> tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")
>>> recognizer = pipeline("ner", model=model, tokenizer=tokenizer)
```

