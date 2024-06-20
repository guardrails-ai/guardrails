# Validators

## ValidLength

Validates that the length of value is within the expected range.

**Key Properties**

| Property                      | Description                       |
| ----------------------------- | --------------------------------- |
| Name for `format` attribute   | `length`                          |
| Supported data types          | `string`, `list`, `object`        |
| Programmatic fix              | If shorter than the minimum, pad with empty last elements. If longer than the maximum, truncate. |

**Arguments**:

- `min` - The inclusive minimum length.
- `max` - The inclusive maximum length.

#### validate(value: Union[str, List], metadata: Dict)

```python
def validate(value: Union[str, List], metadata: Dict) -> ValidationResult
```

Validates that the length of value is within the expected range.

## TwoWords

Validates that a value is two words.

**Key Properties**

| Property                      | Description                       |
| ----------------------------- | --------------------------------- |
| Name for `format` attribute   | `two-words`                       |
| Supported data types          | `string`                          |
| Programmatic fix              | Pick the first two words.         |

## ValidURL

Validates that a value is a valid URL.

**Key Properties**

| Property                      | Description                       |
| ----------------------------- | --------------------------------- |
| Name for `format` attribute   | `valid-url`                       |
| Supported data types          | `string`                          |
| Programmatic fix              | None                              |

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

## ExcludeSqlPredicates

Validates that the SQL query does not contain certain predicates.

**Key Properties**

| Property                      | Description                       |
| ----------------------------- | --------------------------------- |
| Name for `format` attribute   | `exclude-sql-predicates`          |
| Supported data types          | `string`                          |
| Programmatic fix              | None                              |

**Arguments**:

- `predicates` - The list of predicates to avoid.

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

## ExtractedSummarySentencesMatch

Validates that the extracted summary sentences match the original text
by performing a cosine similarity in the embedding space.

**Key Properties**

| Property                      | Description                         |
| ----------------------------- | ----------------------------------- |
| Name for `format` attribute   | `extracted-summary-sentences-match` |
| Supported data types          | `string`                            |
| Programmatic fix              | Remove any sentences that can not be verified. |

**Arguments**:

  
- `threshold` - The minimum cosine similarity to be considered similar. Default to 0.7.
  
  Other parameters: Metadata
  
- `filepaths` _List[str]_ - A list of strings that specifies the filepaths for any documents that should be used for asserting the summary's similarity.
- `document_store` _DocumentStoreBase, optional_ - The document store to use during validation. Defaults to EphemeralDocumentStore.
- `vector_db` _VectorDBBase, optional_ - A vector database to use for embeddings.  Defaults to Faiss.
- `embedding_model` _EmbeddingBase, optional_ - The embeddig model to use. Defaults to OpenAIEmbedding.

## OneLine

Validates that a value is a single line, based on whether or not the
output has a newline character (\n).

**Key Properties**

| Property                      | Description                            |
| ----------------------------- | -------------------------------------- |
| Name for `format` attribute   | `one-line`                             |
| Supported data types          | `string`                               |
| Programmatic fix              | Keep the first line, delete other text |

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

## DetectSecrets

Validates whether the generated code snippet contains any secrets.

**Key Properties**
| Property                      | Description                       |
| ----------------------------- | --------------------------------- |
| Name for `format` attribute   | `detect-secrets`                  |
| Supported data types          | `string`                          |
| Programmatic fix              | None                              |

**Arguments**:

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
  

**Example**:

    ```py

    guard = Guard.from_string(validators=[
        DetectSecrets(on_fail=OnFailAction.FIX)
    ])
    guard.parse(
        llm_output=code_snippet,
    )
    ```

#### get\_unique\_secrets(value: str)

```python
def get_unique_secrets(value: str) -> Tuple[Dict[str, Any], List[str]]
```

Get unique secrets from the value.

**Arguments**:

- `value` _str_ - The generated code snippet.
  

**Returns**:

- `unique_secrets` _Dict[str, Any]_ - A dictionary of unique secrets and their
  line numbers.
- `lines` _List[str]_ - The lines of the generated code snippet.

#### get\_modified\_value(unique\_secrets: Dict[str, Any], lines: List[str])

```python
def get_modified_value(unique_secrets: Dict[str, Any],
                       lines: List[str]) -> str
```

Replace the secrets on the lines with asterisks.

**Arguments**:

- `unique_secrets` _Dict[str, Any]_ - A dictionary of unique secrets and their
  line numbers.
- `lines` _List[str]_ - The lines of the generated code snippet.
  

**Returns**:

- `modified_value` _str_ - The generated code snippet with secrets replaced with
  asterisks.

## PydanticFieldValidator

Validates a specific field in a Pydantic model with the specified
validator method.

**Key Properties**

| Property                      | Description                       |
| ----------------------------- | --------------------------------- |
| Name for `format` attribute   | `pydantic_field_validator`        |
| Supported data types          | `Any`                             |
| Programmatic fix              | Override with return value from `field_validator`.   |

**Arguments**:

  
- `field_validator` _Callable_ - A validator for a specific field in a Pydantic model.

## ValidRange

Validates that a value is within a range.

**Key Properties**

| Property                      | Description                       |
| ----------------------------- | --------------------------------- |
| Name for `format` attribute   | `valid-range`                     |
| Supported data types          | `integer`, `float`, `percentage`  |
| Programmatic fix              | Closest value within the range.   |

**Arguments**:

- `min` - The inclusive minimum value of the range.
- `max` - The inclusive maximum value of the range.

#### validate(value: Any, metadata: Dict)

```python
def validate(value: Any, metadata: Dict) -> ValidationResult
```

Validates that a value is within a range.

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

## EndsWith

Validates that a list ends with a given value.

**Key Properties**

| Property                      | Description                       |
| ----------------------------- | --------------------------------- |
| Name for `format` attribute   | `ends-with`                       |
| Supported data types          | `list`                            |
| Programmatic fix              | Append the given value to the list. |

**Arguments**:

- `end` - The required last element.

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

**Arguments**:

- `valid_topics` _List[str]_ - topics that the text should be about
  (one or many).
- `invalid_topics` _List[str], Optional, defaults to []_ - topics that the
  text cannot be about.
- `device` _int, Optional, defaults to -1_ - Device ordinal for CPU/GPU
  supports for Zero-Shot classifier. Setting this to -1 will leverage
  CPU, a positive will run the Zero-Shot model on the associated CUDA
  device id.
- `model` _str, Optional, defaults to 'facebook/bart-large-mnli'_ - The
  Zero-Shot model that will be used to classify the topic. See a
  list of all models here:
  https://huggingface.co/models?pipeline_tag=zero-shot-classification
  llm_callable (Union[str, Callable, None], Optional, defaults to
- `'gpt-3.5-turbo')` - Either the name of the OpenAI model, or a callable
  that takes a prompt and returns a response.
- `disable_classifier` _bool, Optional, defaults to False_ - controls whether
  to use the Zero-Shot model. At least one of disable_classifier and
  disable_llm must be False.
- `disable_llm` _bool, Optional, defaults to False_ - controls whether to use
  the LLM fallback. At least one of disable_classifier and
  disable_llm must be False.
- `model_threshold` _float, Optional, defaults to 0.5_ - The threshold used to
  determine whether to accept a topic from the Zero-Shot model. Must be
  a number between 0 and 1.

#### call\_llm(text: str, topics: List[str])

```python
@retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(5))
def call_llm(text: str, topics: List[str]) -> str
```

Call the LLM with the given prompt.

Expects a function that takes a string and returns a string.

**Arguments**:

- `text` _str_ - The input text to classify using the LLM.
- `topics` _List[str]_ - The list of candidate topics.

**Returns**:

- `response` _str_ - String representing the LLM response.

#### set\_callable(llm\_callable: Union[str, Callable, None])

```python
def set_callable(llm_callable: Union[str, Callable, None]) -> None
```

Set the LLM callable.

**Arguments**:

- `llm_callable` - Either the name of the OpenAI model, or a callable that takes
  a prompt and returns a response.

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

**Arguments**:

- `pii_entities` _str | List[str], optional_ - The PII entities to filter. Must be
  one of `pii` or `spi`. Defaults to None. Can also be set in metadata.

#### get\_anonymized\_text(text: str, entities: List[str])

```python
def get_anonymized_text(text: str, entities: List[str]) -> str
```

Analyze and anonymize the text for PII.

**Arguments**:

- `text` _str_ - The text to analyze.
- `pii_entities` _List[str]_ - The PII entities to filter.
  

**Returns**:

- `anonymized_text` _str_ - The anonymized text.

## SaliencyCheck

Checks that the summary covers the list of topics present in the
document.

**Key Properties**

| Property                      | Description                         |
| ----------------------------- | ----------------------------------- |
| Name for `format` attribute   | `saliency-check`                    |
| Supported data types          | `string`                            |
| Programmatic fix              | None                                |

**Arguments**:

  
- `docs_dir` - Path to the directory containing the documents.
- `threshold` - Threshold for overlap between topics in document and summary. Defaults to 0.25

#### \_\_init\_\_(docs\_dir: str, llm\_callable: Optional[Callable] = None, on\_fail: Optional[Callable] = None, threshold: float = 0.25, \*\*kwargs)

```python
def __init__(docs_dir: str,
             llm_callable: Optional[Callable] = None,
             on_fail: Optional[Callable] = None,
             threshold: float = 0.25,
             **kwargs)
```

Initialize the SalienceCheck validator.

**Arguments**:

- `docs_dir` - Path to the directory containing the documents.
- `on_fail` - Function to call when validation fails.
- `threshold` - Threshold for overlap between topics in document and summary.

## SqlColumnPresence

Validates that all columns in the SQL query are present in the schema.

**Key Properties**

| Property                      | Description                       |
| ----------------------------- | --------------------------------- |
| Name for `format` attribute   | `sql-column-presence`             |
| Supported data types          | `string`                          |
| Programmatic fix              | None                              |

**Arguments**:

- `cols` - The list of valid columns.

## UpperCase

Validates that a value is upper case.

**Key Properties**

| Property                      | Description                       |
| ----------------------------- | --------------------------------- |
| Name for `format` attribute   | `upper-case`                      |
| Supported data types          | `string`                          |
| Programmatic fix              | Convert to upper case.            |

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

**Arguments**:

  
- `threshold` - The minimum fuzz ratio to be considered redundant.  Defaults to 70.

#### validate(value: Any, metadata: Dict)

```python
def validate(value: Any, metadata: Dict) -> ValidationResult
```

Remove redundant sentences from a string.

## LowerCase

Validates that a value is lower case.

**Key Properties**

| Property                      | Description                       |
| ----------------------------- | --------------------------------- |
| Name for `format` attribute   | `lower-case`                      |
| Supported data types          | `string`                          |
| Programmatic fix              | Convert to lower case.            |

## SimilarToList

Validates that a value is similar to a list of previously known values.

**Key Properties**

| Property                      | Description                       |
| ----------------------------- | --------------------------------- |
| Name for `format` attribute   | `similar-to-list`                 |
| Supported data types          | `string`                          |
| Programmatic fix              | None                              |

**Arguments**:

- `standard_deviations` _int_ - The number of standard deviations from the mean to check.
- `threshold` _float_ - The threshold for the average semantic similarity for strings.
  
  For integer values, this validator checks whether the value lies
  within 'k' standard deviations of the mean of the previous values.
  (Assumes that the previous values are normally distributed.) For
  string values, this validator checks whether the average semantic
  similarity between the generated value and the previous values is
  less than a threshold.

#### get\_semantic\_similarity(text1: str, text2: str, embed\_function: Callable)

```python
def get_semantic_similarity(text1: str, text2: str,
                            embed_function: Callable) -> float
```

Get the semantic similarity between two strings.

**Arguments**:

- `text1` _str_ - The first string.
- `text2` _str_ - The second string.
- `embed_function` _Callable_ - The embedding function.

**Returns**:

- `similarity` _float_ - The semantic similarity between the two strings.

## ProvenanceV0

Validates that LLM-generated text matches some source text based on
distance in embedding space.

**Key Properties**

| Property                      | Description                         |
| ----------------------------- | ----------------------------------- |
| Name for `format` attribute   | `provenance-v0`                     |
| Supported data types          | `string`                            |
| Programmatic fix              | None                                |

**Arguments**:

- `threshold` - The minimum cosine similarity between the generated text and
  the source text. Defaults to 0.8.
- `validation_method` - Whether to validate at the sentence level or over the full text.  Must be one of `sentence` or `full`. Defaults to `sentence`
  
  Other parameters: Metadata
- `query_function` _Callable, optional_ - A callable that takes a string and returns a list of (chunk, score) tuples.
- `sources` _List[str], optional_ - The source text.
- `embed_function` _Callable, optional_ - A callable that creates embeddings for the sources. Must accept a list of strings and return an np.array of floats.
  
  In order to use this validator, you must provide either a `query_function` or
  `sources` with an `embed_function` in the metadata.
  
  If providing query_function, it should take a string as input and return a list of
  (chunk, score) tuples. The chunk is a string and the score is a float representing
  the cosine distance between the chunk and the input string. The list should be
  sorted in ascending order by score.
  
- `Note` - The score should represent distance in embedding space, not similarity. I.e.,
  lower is better and the score should be 0 if the chunk is identical to the input
  string.
  

**Example**:

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
  

**Example**:

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


Example using a custom llm callable:


OR

2. `sources` with an `embed_function` in the metadata. The embed_function should
take a string or a list of strings as input and return a np array of floats.
The vector should be normalized to unit length.

``` py
def query_function(text: str, k: int) -> List[str]:
    return ["This is a chunk", "This is another chunk"]

guard = Guard.from_string(validators=[
    ProvenanceV1(llm_callable="gpt-3.5-turbo", ...)
])
guard.parse(
    llm_output=...,
    metadata={"query_function": query_function}
)
```
``` py
def query_function(text: str, k: int) -> List[str]:
    return ["This is a chunk", "This is another chunk"]

guard = Guard.from_string(validators=[
        ProvenanceV1(llm_callable=your_custom_callable, ...)
    ]
)
guard.parse(
    llm_output=...,
    metadata={"query_function": query_function}
)
```

**Example**:

  
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

#### \_\_init\_\_(validation\_method: str = "sentence", llm\_callable: Union[str, Callable] = "gpt-3.5-turbo", top\_k: int = 3, max\_tokens: int = 2, on\_fail: Optional[Callable] = None, \*\*kwargs)

```python
def __init__(validation_method: str = "sentence",
             llm_callable: Union[str, Callable] = "gpt-3.5-turbo",
             top_k: int = 3,
             max_tokens: int = 2,
             on_fail: Optional[Callable] = None,
             **kwargs)
```

args:
validation_method (str): Whether to validate at the sentence level or over
the full text.  One of `sentence` or `full`. Defaults to `sentence`
llm_callable (Union[str, Callable]): Either the name of the OpenAI model,
or a callable that takes a prompt and returns a response.
top_k (int): The number of chunks to return from the query function.
Defaults to 3.
max_tokens (int): The maximum number of tokens to send to the LLM.
Defaults to 2.

Other args: Metadata
query_function (Callable): A callable that takes a string and returns a
list of chunks.
sources (List[str], optional): The source text.
embed_function (Callable, optional): A callable that creates embeddings for
the sources. Must accept a list of strings and returns float np.array.

#### set\_callable(llm\_callable: Union[str, Callable])

```python
def set_callable(llm_callable: Union[str, Callable]) -> None
```

Set the LLM callable.

**Arguments**:

- `llm_callable` - Either the name of the OpenAI model, or a callable that takes
  a prompt and returns a response.

#### call\_llm(prompt: str)

```python
@retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(6))
def call_llm(prompt: str) -> str
```

Call the LLM with the given prompt.

Expects a function that takes a string and returns a string.

**Arguments**:

- `prompt` _str_ - The prompt to send to the LLM.
  

**Returns**:

- `response` _str_ - String representing the LLM response.

#### evaluate\_with\_llm(text: str, query\_function: Callable)

```python
def evaluate_with_llm(text: str, query_function: Callable) -> bool
```

Validate that the LLM-generated text is supported by the provided
contexts.

**Arguments**:

- `value` _Any_ - The LLM-generated text.
- `query_function` _Callable_ - The query function.
  

**Returns**:

- `self_eval` - The self-evaluation boolean

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

## CompetitorCheck

Validates that LLM-generated text is not naming any competitors from a
given list.

In order to use this validator you need to provide an extensive list of the
competitors you want to avoid naming including all common variations.

**Arguments**:

- `competitors` _List[str]_ - List of competitors you want to avoid naming

#### exact\_match(text: str, competitors: List[str])

```python
def exact_match(text: str, competitors: List[str]) -> List[str]
```

Performs exact match to find competitors from a list in a given
text.

**Arguments**:

- `text` _str_ - The text to search for competitors.
- `competitors` _list_ - A list of competitor entities to match.
  

**Returns**:

- `list` - A list of matched entities.

#### perform\_ner(text: str, nlp)

```python
def perform_ner(text: str, nlp) -> List[str]
```

Performs named entity recognition on text using a provided NLP
model.

**Arguments**:

- `text` _str_ - The text to perform named entity recognition on.
- `nlp` - The NLP model to use for entity recognition.
  

**Returns**:

- `entities` - A list of entities found.

#### is\_entity\_in\_list(entities: List[str], competitors: List[str])

```python
def is_entity_in_list(entities: List[str], competitors: List[str]) -> List
```

Checks if any entity from a list is present in a given list of
competitors.

**Arguments**:

- `entities` _list_ - A list of entities to check
- `competitors` _list_ - A list of competitor names to match
  

**Returns**:

- `List` - List of found competitors

#### validate(value: str, metadata=Dict)

```python
def validate(value: str, metadata=Dict) -> ValidationResult
```

Checks a text to find competitors' names in it.

While running, store sentences naming competitors and generate a fixed output
filtering out all flagged sentences.

**Arguments**:

- `value` _str_ - The value to be validated.
- `metadata` _Dict, optional_ - Additional metadata. Defaults to empty dict.
  

**Returns**:

- `ValidationResult` - The validation result.

## RegexMatch

Validates that a value matches a regular expression.

**Key Properties**

| Property                      | Description                       |
| ----------------------------- | --------------------------------- |
| Name for `format` attribute   | `regex_match`                     |
| Supported data types          | `string`                          |
| Programmatic fix              | Generate a string that matches the regular expression |

**Arguments**:

- `regex` - Str regex pattern
- `match_type` - Str in {"search", "fullmatch"} for a regex search or full-match option

## ToxicLanguage

Validates that the generated text is toxic.

**Key Properties**
| Property                      | Description                       |
| ----------------------------- | --------------------------------- |
| Name for `format` attribute   | `toxic-language`                  |
| Supported data types          | `string`                          |
| Programmatic fix              | None                              |

**Arguments**:

- `threshold` - The confidence threshold (model inference) for toxicity.
  Defaults to 0.5.
- `validation_method` - Whether to validate at the sentence level or
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

#### get\_toxicity(value: str)

```python
def get_toxicity(value: str) -> List[str]
```

Check whether the generated text is toxic.

Returns the labels predicted by the model with
confidence higher than the threshold.

**Arguments**:

- `value` _str_ - The generated text.
  

**Returns**:

- `pred_labels` _bool_ - Labels predicted by the model
  with confidence higher than the threshold.

#### validate\_each\_sentence(value: str, metadata: Dict[str, Any])

```python
def validate_each_sentence(value: str,
                           metadata: Dict[str, Any]) -> ValidationResult
```

Validate that each sentence in the generated text is toxic.

#### validate\_full\_text(value: str, metadata: Dict[str, Any])

```python
def validate_full_text(value: str, metadata: Dict[str,
                                                  Any]) -> ValidationResult
```

Validate that the entire generated text is toxic.

## ValidChoices

Validates that a value is within the acceptable choices.

**Key Properties**

| Property                      | Description                       |
| ----------------------------- | --------------------------------- |
| Name for `format` attribute   | `valid-choices`                   |
| Supported data types          | `all`                             |
| Programmatic fix              | None                              |

**Arguments**:

- `choices` - The list of valid choices.

#### validate(value: Any, metadata: Dict)

```python
def validate(value: Any, metadata: Dict) -> ValidationResult
```

Validates that a value is within a range.

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

**Arguments**:

- `document` - The document to use for the similarity check.
- `threshold` - The minimum cosine similarity to be considered similar.  Defaults to 0.7.
- `model` - The embedding model to use.  Defaults to text-embedding-ada-002.

#### cosine\_similarity(a: "np.ndarray", b: "np.ndarray")

```python
@staticmethod
def cosine_similarity(a: "np.ndarray", b: "np.ndarray") -> float
```

Calculate the cosine similarity between two vectors.

**Arguments**:

- `a` - The first vector.
- `b` - The second vector.
  

**Returns**:

- `float` - The cosine similarity between the two vectors.

## ReadingTime

Validates that the a string can be read in less than a certain amount of
time.

**Key Properties**

| Property                      | Description                         |
| ----------------------------- | ----------------------------------- |
| Name for `format` attribute   | `reading-time`                      |
| Supported data types          | `string`                            |
| Programmatic fix              | None                                |

**Arguments**:

  
- `reading_time` - The maximum reading time in minutes.

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

**Arguments**:

  
- `threshold` - The minimum fuzz ratio to be considered summarized.  Defaults to 85.
  
  Other parameters: Metadata
  
- `filepaths` _List[str]_ - A list of strings that specifies the filepaths for any documents that should be used for asserting the summary's similarity.

#### validate(value: Any, metadata: Dict)

```python
def validate(value: Any, metadata: Dict) -> ValidationResult
```

Make sure each sentence was precisely copied from the document.

## EndpointIsReachable

Validates that a value is a reachable URL.

**Key Properties**

| Property                      | Description                       |
| ----------------------------- | --------------------------------- |
| Name for `format` attribute   | `is-reachable`                    |
| Supported data types          | `string`,                         |
| Programmatic fix              | None                              |

