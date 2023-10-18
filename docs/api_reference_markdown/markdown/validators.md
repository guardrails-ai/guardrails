# Validators





































































































































































































































































































































































































































































































































































































  |



























































































































































































































































































































































































































































































































































































 | Property                  | Description   |
|---------------------------|---------------|
| Name for format attribute | bug-free-sql  |
| Supported data types      | sql, string   |
| Programmatic fix          | None          |

#### \_\_init_\_(conn=None, schema_file=None, on_fail=None)

* **Parameters:**
* **conn** (*str* *|* *None*) – 
* **schema_file** (*str* *|* *None*) – 
* **on_fail** (*Callable* *|* *None*) – 

### *class* guardrails.validators.EndpointIsReachable

Validates that a value is a reachable URL.

**Key Properties**

| Property                  | Description   |
|---------------------------|---------------|
| Name for format attribute | is-reachable  |
| Supported data types      | string, url   |
| Programmatic fix          | None          |

### *class* guardrails.validators.EndsWith

Validates that a list ends with a given value.

**Key Properties**

| Property                  | Description                             |
|---------------------------|-----------------------------------------|
| Name for format attribute | ends-with                               |
| Supported data types      | list                                    |
| Programmatic fix          | Append the given value to the&lt;br/>list. |
Parameters: Arguments
: end: The required last element.

#### \_\_init_\_(end, on_fail='fix')

* **Parameters:**
* **end** (*str*) – 
* **on_fail** (*str*) – 

### *class* guardrails.validators.ExcludeSqlPredicates

Validates that the SQL query does not contain certain predicates.

**Key Properties**

| Property                  | Description            |
|---------------------------|------------------------|
| Name for format attribute | exclude-sql-predicates |
| Supported data types      | sql                    |
| Programmatic fix          | None                   |
Parameters: Arguments
: predicates: The list of predicates to avoid.

#### \_\_init_\_(predicates, on_fail=None)

* **Parameters:**
* **predicates** (*List**[**str**]*) – 
* **on_fail** (*Callable* *|* *None*) – 

### *class* guardrails.validators.ExtractedSummarySentencesMatch

Validates that the extracted summary sentences match the original text
by performing a cosine similarity in the embedding space.

**Key Properties**

| Property                  | Description                                        |
|---------------------------|----------------------------------------------------|
| Name for format attribute | extracted-summary-sentences-match                  |
| Supported data types      | string                                             |
| Programmatic fix          | Remove any sentences that can not be&lt;br/>verified. |

Parameters: Arguments

 threshold: The minimum cosine similarity to be considered similar. Default to 0.7.

Other parameters: Metadata

 filepaths (List[str]): A list of strings that specifies the filepaths for any documents that should be used for asserting the summary’s similarity.
 document_store (DocumentStoreBase, optional): The document store to use during validation. Defaults to EphemeralDocumentStore.
 vector_db (VectorDBBase, optional): A vector database to use for embeddings.  Defaults to Faiss.
 embedding_model (EmbeddingBase, optional): The embeddig model to use. Defaults to OpenAIEmbedding.

#### \_\_init_\_(threshold=0.7, on_fail=None, \*\*kwargs)

* **Parameters:**
* **threshold** (*float*) – 
* **on_fail** (*Callable* *|* *None*) – 
* **kwargs** (*Dict**[**str**,* *Any**]* *|* *None*) – 

#### to_prompt(with_keywords=True)

Convert the validator to a prompt.

E.g. ValidLength(5, 10) -> “length: 5 10” when with_keywords is False.
ValidLength(5, 10) -> “length: min=5 max=10” when with_keywords is True.

* **Parameters:**
**with_keywords** (*bool*) – Whether to include the keyword arguments in the prompt.
* **Returns:**
A string representation of the validator.
* **Return type:**
str

### *class* guardrails.validators.ExtractiveSummary

Validates that a string is a valid extractive summary of a given
document.

This validator does a fuzzy match between the sentences in the
summary and the sentences in the document. Each sentence in the
summary must be similar to at least one sentence in the document.
After the validation, the summary is updated to include the
sentences from the document that were matched, and the citations for
those sentences are added to the end of the summary.

**Key Properties**

| Property                  | Description                                        |
|---------------------------|----------------------------------------------------|
| Name for format attribute | extractive-summary                                 |
| Supported data types      | string                                             |
| Programmatic fix          | Remove any sentences that can not&lt;br/>be verified. |

Parameters: Arguments

 threshold: The minimum fuzz ratio to be considered summarized.  Defaults to 85.

Other parameters: Metadata

 filepaths (List[str]): A list of strings that specifies the filepaths for any documents that should be used for asserting the summary’s similarity.

#### \_\_init_\_(threshold=85, on_fail=None, \*\*kwargs)

* **Parameters:**
* **threshold** (*int*) – 
* **on_fail** (*Callable* *|* *None*) – 

### *class* guardrails.validators.IsHighQualityTranslation

Using inpiredco.critique to check if a translation is high quality.

**Key Properties**

| Property                  | Description                 |
|---------------------------|-----------------------------|
| Name for format attribute | is-high-quality-translation |
| Supported data types      | string                      |
| Programmatic fix          | None                        |
Other parameters: Metadata
: translation_source (str): The source of the translation.

#### \_\_init_\_(\*args, \*\*kwargs)

### *class* guardrails.validators.IsProfanityFree

Validates that a translated text does not contain profanity language.

This validator uses the alt-profanity-check package to check if a string
contains profanity language.

**Key Properties**

| Property                  | Description       |
|---------------------------|-------------------|
| Name for format attribute | is-profanity-free |
| Supported data types      | string            |
| Programmatic fix          | None              |

### *class* guardrails.validators.LowerCase

Validates that a value is lower case.

**Key Properties**

| Property                  | Description            |
|---------------------------|------------------------|
| Name for format attribute | lower-case             |
| Supported data types      | string                 |
| Programmatic fix          | Convert to lower case. |

### *class* guardrails.validators.OneLine

Validates that a value is a single line or sentence.

**Key Properties**

| Property                  | Description          |
|---------------------------|----------------------|
| Name for format attribute | one-line             |
| Supported data types      | string               |
| Programmatic fix          | Pick the first line. |

### *class* guardrails.validators.ProvenanceV0

Validates that LLM-generated text matches some source text based on
distance in embedding space.

**Key Properties**

| Property                  | Description   |
|---------------------------|---------------|
| Name for format attribute | provenance-v0 |
| Supported data types      | string        |
| Programmatic fix          | None          |
Parameters: Arguments
: threshold: The minimum cosine similarity between the generated text and
: the source text. Defaults to 0.8.
&lt;br/>
validation_method: Whether to validate at the sentence level or over the full text.
: Must be one of sentence or full. Defaults to sentence

Other parameters: Metadata
: query_function (Callable, optional): A callable that takes a string and returns
: a list of (chunk, score) tuples.
&lt;br/>
sources (List[str], optional): The source text.
embed_function (Callable, optional): A callable that creates embeddings for the
&lt;br/>
 sources. Must accept a list of strings and return an np.array of floats.

In order to use this validator, you must provide either a query_function or
sources with an embed_function in the metadata.

If providing query_function, it should take a string as input and return a list of
(chunk, score) tuples. The chunk is a string and the score is a float representing
the cosine similarity between the chunk and the input string. The list should be
sorted in ascending order by score.

Example:

```default
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

```default
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

#### \_\_init_\_(threshold=0.8, validation_method='sentence', on_fail=None, \*\*kwargs)

* **Parameters:**
* **threshold** (*float*) – 
* **validation_method** (*str*) – 
* **on_fail** (*Callable* *|* *None*) – 

#### to_prompt(with_keywords=True)

Convert the validator to a prompt.

E.g. ValidLength(5, 10) -> “length: 5 10” when with_keywords is False.
ValidLength(5, 10) -> “length: min=5 max=10” when with_keywords is True.

* **Parameters:**
**with_keywords** (*bool*) – Whether to include the keyword arguments in the prompt.
* **Returns:**
A string representation of the validator.
* **Return type:**
str

### *class* guardrails.validators.ProvenanceV1

Validates that the LLM-generated text is supported by the provided
contexts.

This validator uses an LLM callable to evaluate the generated text against the
provided contexts (LLM-ception).

In order to use this validator, you must provide either:
1. a ‘query_function’ in the metadata. That function should take a string as input

 (the LLM-generated text) and return a list of relevant
chunks. The list should be sorted in ascending order by the distance between the
: chunk and the LLM-generated text.

Example using str callable:
: ```pycon
>> def query_function(text: str, k: int) -> List[str]:
...     return ["This is a chunk", "This is another chunk"]
```
&lt;br/>
```pycon
>> guard = Guard.from_string(validators=[
ProvenanceV1(llm_callable="gpt-3.5-turbo", ...)
]
)
>> guard.parse(
...   llm_output=...,
...   metadata={"query_function": query_function}
... )
```

Example using a custom llm callable:
: ```pycon
>> def query_function(text: str, k: int) -> List[str]:
...     return ["This is a chunk", "This is another chunk"]
```
&lt;br/>
```pycon
>> guard = Guard.from_string(validators=[
ProvenanceV1(llm_callable=your_custom_callable, ...)
]
)
>> guard.parse(
...   llm_output=...,
...   metadata={"query_function": query_function}
... )
```

OR

1. sources with an embed_function in the metadata. The embed_function should
: take a string or a list of strings as input and return a np array of floats.

The vector should be normalized to unit length.

Example:

```default
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

#### \_\_init_\_(validation_method='sentence', llm_callable='gpt-3.5-turbo', top_k=3, max_tokens=2, on_fail=None, \*\*kwargs)

* **Parameters:**
* **validation_method** (`str`) – Whether to validate at the sentence level or over
the full text.  One of sentence or full. Defaults to sentence
* **llm_callable** (`Union[str, Callable]`) – Either the name of the OpenAI model,
or a callable that takes a prompt and returns a response.
* **top_k** (`int`) – The number of chunks to return from the query function.
Defaults to 3.
* **max_tokens** (`int`) – The maximum number of tokens to send to the LLM.
Defaults to 2.
* **on_fail** (*Callable* *|* *None*) – 

Other args: Metadata
: query_function (Callable): A callable that takes a string and returns a
: list of chunks.
&lt;br/>
sources (List[str], optional): The source text.
embed_function (Callable, optional): A callable that creates embeddings for
&lt;br/>
 the sources. Must accept a list of strings and returns float np.array.

#### call_llm(prompt)

Call the LLM with the given prompt.

Expects a function that takes a string and returns a string.

* **Parameters:**
**prompt** (`str`) – The prompt to send to the LLM.
* **Returns:**
String representing the LLM response.
* **Return type:**
response (str)

#### evaluate_with_llm(text, query_function)

Validate that the LLM-generated text is supported by the provided
contexts.

* **Parameters:**
* **value** (`Any`) – The LLM-generated text.
* **query_function** (`Callable`) – The query function.
* **text** (*str*) – 
* **Returns:**
The self-evaluation boolean
* **Return type:**
self_eval

#### set_callable(llm_callable)

Set the LLM callable.

* **Parameters:**
**llm_callable** (*str* *|* *Callable*) – Either the name of the OpenAI model, or a callable that takes
a prompt and returns a response.
* **Return type:**
None

### *class* guardrails.validators.QARelevanceLLMEval

Validates that an answer is relevant to the question asked by asking the
LLM to self evaluate.

**Key Properties**

| Property                  | Description           |
|---------------------------|-----------------------|
| Name for format attribute | qa-relevance-llm-eval |
| Supported data types      | string                |
| Programmatic fix          | None                  |
Other parameters: Metadata
: question (str): The original question the llm was given to answer.

#### \_\_init_\_(llm_callable=None, on_fail=None, \*\*kwargs)

* **Parameters:**
* **llm_callable** (*Callable* *|* *None*) – 
* **on_fail** (*Callable* *|* *None*) – 

#### to_prompt(with_keywords=True)

Convert the validator to a prompt.

E.g. ValidLength(5, 10) -> “length: 5 10” when with_keywords is False.
ValidLength(5, 10) -> “length: min=5 max=10” when with_keywords is True.

* **Parameters:**
**with_keywords** (*bool*) – Whether to include the keyword arguments in the prompt.
* **Returns:**
A string representation of the validator.
* **Return type:**
str

### *class* guardrails.validators.ReadingTime

Validates that the a string can be read in less than a certain amount of
time.

**Key Properties**

| Property                  | Description   |
|---------------------------|---------------|
| Name for format attribute | reading-time  |
| Supported data types      | string        |
| Programmatic fix          | None          |

Parameters: Arguments

 reading_time: The maximum reading time.

#### \_\_init_\_(reading_time, on_fail='fix')

* **Parameters:**
* **reading_time** (*int*) – 
* **on_fail** (*str*) – 

### *class* guardrails.validators.RemoveRedundantSentences

Removes redundant sentences from a string.

This validator removes sentences from a string that are similar to
other sentences in the string. This is useful for removing
repetitive sentences from a string.

**Key Properties**

| Property                  | Description                     |
|---------------------------|---------------------------------|
| Name for format attribute | remove-redundant-sentences      |
| Supported data types      | string                          |
| Programmatic fix          | Remove any redundant sentences. |

Parameters: Arguments

 threshold: The minimum fuzz ratio to be considered redundant.  Defaults to 70.

#### \_\_init_\_(threshold=70, on_fail=None, \*\*kwargs)

* **Parameters:**
* **threshold** (*int*) – 
* **on_fail** (*Callable* *|* *None*) – 

### *class* guardrails.validators.SaliencyCheck

Checks that the summary covers the list of topics present in the
document.

**Key Properties**

| Property                  | Description    |
|---------------------------|----------------|
| Name for format attribute | saliency-check |
| Supported data types      | string         |
| Programmatic fix          | None           |

Parameters: Arguments

 docs_dir: Path to the directory containing the documents.
 threshold: Threshold for overlap between topics in document and summary. Defaults to 0.25

#### \_\_init_\_(docs_dir, llm_callable=None, on_fail=None, threshold=0.25, \*\*kwargs)

Initialize the SalienceCheck validator.

* **Parameters:**
* **docs_dir** (*str*) – Path to the directory containing the documents.
* **on_fail** (*Callable* *|* *None*) – Function to call when validation fails.
* **threshold** (*float*) – Threshold for overlap between topics in document and summary.
* **llm_callable** (*Callable* *|* *None*) – 

#### \_get_topics(text, topics=None)

Extract topics from a string.

* **Parameters:**
* **text** (*str*) – 
* **topics** (*List**[**str**]* *|* *None*) – 
* **Return type:**
*List*[str]

#### *property* \_topics*: List[str]*

Return a list of topics that can be used in the validator.

### *class* guardrails.validators.SimilarToDocument

Validates that a value is similar to the document.

This validator checks if the value is similar to the document by checking
the cosine similarity between the value and the document, using an
embedding.

**Key Properties**

| Property                  | Description         |
|---------------------------|---------------------|
| Name for format attribute | similar-to-document |
| Supported data types      | string              |
| Programmatic fix          | None                |
Parameters: Arguments
: document: The document to use for the similarity check.
threshold: The minimum cosine similarity to be considered similar.  Defaults to 0.7.
model: The embedding model to use.  Defaults to text-embedding-ada-002.

#### \_\_init_\_(document, threshold=0.7, model='text-embedding-ada-002', on_fail=None)

* **Parameters:**
* **document** (*str*) – 
* **threshold** (*float*) – 
* **model** (*str*) – 
* **on_fail** (*Callable* *|* *None*) – 

#### *static* cosine_similarity(a, b)

Calculate the cosine similarity between two vectors.

* **Parameters:**
* **a** (*ndarray*) – The first vector.
* **b** (*ndarray*) – The second vector.
* **Returns:**
The cosine similarity between the two vectors.
* **Return type:**
float

#### to_prompt(with_keywords=True)

Convert the validator to a prompt.

E.g. ValidLength(5, 10) -> “length: 5 10” when with_keywords is False.
ValidLength(5, 10) -> “length: min=5 max=10” when with_keywords is True.

* **Parameters:**
**with_keywords** (*bool*) – Whether to include the keyword arguments in the prompt.
* **Returns:**
A string representation of the validator.
* **Return type:**
str

### *class* guardrails.validators.SqlColumnPresence

Validates that all columns in the SQL query are present in the schema.

**Key Properties**

| Property                  | Description         |
|---------------------------|---------------------|
| Name for format attribute | sql-column-presence |
| Supported data types      | sql                 |
| Programmatic fix          | None                |
Parameters: Arguments
: cols: The list of valid columns.

#### \_\_init_\_(cols, on_fail=None)

* **Parameters:**
* **cols** (*List**[**str**]*) – 
* **on_fail** (*Callable* *|* *None*) – 

### *class* guardrails.validators.TwoWords

Validates that a value is two words.

**Key Properties**

| Property                  | Description               |
|---------------------------|---------------------------|
| Name for format attribute | two-words                 |
| Supported data types      | string                    |
| Programmatic fix          | Pick the first two words. |

### *class* guardrails.validators.UpperCase

Validates that a value is upper case.

**Key Properties**

| Property                  | Description            |
|---------------------------|------------------------|
| Name for format attribute | upper-case             |
| Supported data types      | string                 |
| Programmatic fix          | Convert to upper case. |

### *class* guardrails.validators.ValidChoices

Validates that a value is within the acceptable choices.

**Key Properties**

| Property                  | Description   |
|---------------------------|---------------|
| Name for format attribute | valid-choices |
| Supported data types      | all           |
| Programmatic fix          | None          |
Parameters: Arguments
: choices: The list of valid choices.

#### \_\_init_\_(choices, on_fail=None)

* **Parameters:**
* **choices** (*List**[**Any**]*) – 
* **on_fail** (*Callable* *|* *None*) – 

### *class* guardrails.validators.ValidLength

Validates that the length of value is within the expected range.

**Key Properties**

| Property                  | Description                                                                                                     |
|---------------------------|-----------------------------------------------------------------------------------------------------------------|
| Name for format attribute | length                                                                                                          |
| Supported data types      | string, list, object                                                                                            |
| Programmatic fix          | If shorter than the minimum,&lt;br/>: pad with empty last elements.&lt;br/>&lt;br/>If longer than the maximum, truncate. |
Parameters: Arguments
: min: The inclusive minimum length.
max: The inclusive maximum length.

#### \_\_init_\_(min=None, max=None, on_fail=None)

* **Parameters:**
* **min** (*int* *|* *None*) – 
* **max** (*int* *|* *None*) – 
* **on_fail** (*Callable* *|* *None*) – 

### *class* guardrails.validators.ValidRange

Validates that a value is within a range.

**Key Properties**

| Property                  | Description                     |
|---------------------------|---------------------------------|
| Name for format attribute | valid-range                     |
| Supported data types      | integer, float, percentage      |
| Programmatic fix          | Closest value within the range. |
Parameters: Arguments
: min: The inclusive minimum value of the range.
max: The inclusive maximum value of the range.

#### \_\_init_\_(min=None, max=None, on_fail=None)

* **Parameters:**
* **min** (*int* *|* *None*) – 
* **max** (*int* *|* *None*) – 
* **on_fail** (*Callable* *|* *None*) – 

### *class* guardrails.validators.ValidURL

Validates that a value is a valid URL.

**Key Properties**

| Property                  | Description   |
|---------------------------|---------------|
| Name for format attribute | valid-url     |
| Supported data types      | string, url   |
| Programmatic fix          | None          |

### *exception* guardrails.validators.ValidatorError

Base class for all validator errors.

### guardrails.validators.check_refrain_in_dict(schema)

Checks if a Refrain object exists in a dict.

* **Parameters:**
**schema** (*Dict*) – A dict that can contain lists, dicts or scalars.
* **Returns:**
True if a Refrain object exists in the dict.
* **Return type:**
bool

### guardrails.validators.check_refrain_in_list(schema)

Checks if a Refrain object exists in a list.

* **Parameters:**
**schema** (*List*) – A list that can contain lists, dicts or scalars.
* **Returns:**
True if a Refrain object exists in the list.
* **Return type:**
bool

### guardrails.validators.filter_in_dict(schema)

Remove out all Filter objects from a dictionary.

* **Parameters:**
**schema** (*Dict*) – A dictionary that can contain lists, dicts or scalars.
* **Returns:**
A dictionary with all Filter objects removed.
* **Return type:**
*Dict*

### guardrails.validators.filter_in_list(schema)

Remove out all Filter objects from a list.

* **Parameters:**
**schema** (*List*) – A list that can contain lists, dicts or scalars.
* **Returns:**
A list with all Filter objects removed.
* **Return type:**
*List*


> | Property                  | Description   |
> |---------------------------|---------------|
> | Name for format attribute | valid-url     |
> | Supported data types      | string, url   |
> | Programmatic fix          | None          |

> ### *exception* guardrails.validators.ValidatorError

> Base class for all validator errors.

> ### guardrails.validators.check_refrain_in_dict(schema)

> Checks if a Refrain object exists in a dict.

> * **Parameters:**
>   **schema** (*Dict*) – A dict that can contain lists, dicts or scalars.
> * **Returns:**
>   True if a Refrain object exists in the dict.
> * **Return type:**
>   bool

> ### guardrails.validators.check_refrain_in_list(schema)

> Checks if a Refrain object exists in a list.

> * **Parameters:**
>   **schema** (*List*) – A list that can contain lists, dicts or scalars.
> * **Returns:**
>   True if a Refrain object exists in the list.
> * **Return type:**
>   bool

> ### guardrails.validators.filter_in_dict(schema)

> Remove out all Filter objects from a dictionary.

> * **Parameters:**
>   **schema** (*Dict*) – A dictionary that can contain lists, dicts or scalars.
> * **Returns:**
>   A dictionary with all Filter objects removed.
> * **Return type:**
>   *Dict*

> ### guardrails.validators.filter_in_list(schema)

> Remove out all Filter objects from a list.

> * **Parameters:**
>   **schema** (*List*) – A list that can contain lists, dicts or scalars.
> * **Returns:**
>   A list with all Filter objects removed.
> * **Return type:**
>   *List*
