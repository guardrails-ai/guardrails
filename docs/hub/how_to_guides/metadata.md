# Passing metadata to the Validators

## Why do some validators require metadata?

Some validators require additional metadata at runtime to perform their validation. For example, the `Provenance_LLM` validator at runtime requires the `filepaths` key in the metadata. This is because the `Provenance_LLM` checks for hallucinations in RAG workflows by comparing the generated output to the source documents. The `filepaths` key in the metadata is used to specify which documents to use for the comparison.


## Passing metadata for a single validator

In order to pass metadata to a validator, you can pass the metadata dictionary to the `guard.validate` method. For example, below is an example for passing metadata to the `Provenance_LLM` validator.

First, we set up the validator and the guard.

```python
# Import Guard and Validator
from guardrails.hub import ProvenanceLLM
from guardrails import Guard

# Import embedding model
from sentence_transformers import SentenceTransformer

# Initialize Validator
val = ProvenanceLLM(
    validation_method="sentence",
    llm_callable="gpt-3.5-turbo",
    top_k=3,
    max_tokens=2,
)

# Setup Guard
guard = Guard.from_string(validators=[val])
```

Below, we show how to pass the metadata to the `guard.validate` method.

```python
# Setup text sources
sources = [
    "The sun is a star.",
    "The sun rises in the east and sets in the west."
]

# Load model for embedding function
model = SentenceTransformer('paraphrase-MiniLM-L6-v2')

# Create embed function
def embed_function(sources: list[str]) -> np.array:
    return model.encode(sources)

guard.validate(
    llm_output="The sun rises in the east.",
    metadata={
        "sources": sources,
        "embed_function": embed_function
    }
)
```


## Passing metadata for multiple validators