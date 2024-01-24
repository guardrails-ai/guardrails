import contextvars
import itertools
import os
import warnings
from functools import partial
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

from tenacity import retry, stop_after_attempt, wait_random_exponential

from guardrails.utils.docs_utils import get_chunks_from_text
from guardrails.utils.openai_utils import OpenAIClient
from guardrails.utils.validator_utils import PROVENANCE_V1_PROMPT
from guardrails.validator_base import (
    FailResult,
    PassResult,
    ValidationResult,
    Validator,
    register_validator,
)

try:
    import numpy as np
except ImportError:
    _HAS_NUMPY = False
else:
    _HAS_NUMPY = True

try:
    import nltk  # type: ignore
except ImportError:
    nltk = None  # type: ignore

if nltk is not None:
    try:
        nltk.data.find("tokenizers/punkt")
    except LookupError:
        nltk.download("punkt")


@register_validator(name="provenance-v0", data_type="string")
class ProvenanceV0(Validator):
    """Validates that LLM-generated text matches some source text based on
    distance in embedding space.

    **Key Properties**

    | Property                      | Description                         |
    | ----------------------------- | ----------------------------------- |
    | Name for `format` attribute   | `provenance-v0`                     |
    | Supported data types          | `string`                            |
    | Programmatic fix              | None                                |

    Args:
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
    """  # noqa

    def __init__(
        self,
        threshold: float = 0.8,
        validation_method: str = "sentence",
        on_fail: Optional[Callable] = None,
        **kwargs,
    ):
        super().__init__(
            on_fail, threshold=threshold, validation_method=validation_method, **kwargs
        )
        self._threshold = float(threshold)
        if validation_method not in ["sentence", "full"]:
            raise ValueError("validation_method must be 'sentence' or 'full'.")
        self._validation_method = validation_method

    def get_query_function(self, metadata: Dict[str, Any]) -> Callable:
        query_fn = metadata.get("query_function", None)
        sources = metadata.get("sources", None)

        # Check that query_fn or sources are provided
        if query_fn is not None:
            if sources is not None:
                warnings.warn(
                    "Both `query_function` and `sources` are provided in metadata. "
                    "`query_function` will be used."
                )
            return query_fn

        if sources is None:
            raise ValueError(
                "You must provide either `query_function` or `sources` in metadata."
            )

        # Check chunking strategy
        chunk_strategy = metadata.get("chunk_strategy", "sentence")
        if chunk_strategy not in ["sentence", "word", "char", "token"]:
            raise ValueError(
                "`chunk_strategy` must be one of 'sentence', 'word', 'char', "
                "or 'token'."
            )
        chunk_size = metadata.get("chunk_size", 5)
        chunk_overlap = metadata.get("chunk_overlap", 2)

        # Check distance metric
        distance_metric = metadata.get("distance_metric", "cosine")
        if distance_metric not in ["cosine", "euclidean"]:
            raise ValueError(
                "`distance_metric` must be one of 'cosine' or 'euclidean'."
            )

        # Check embed model
        embed_function = metadata.get("embed_function", None)
        if embed_function is None:
            raise ValueError(
                "You must provide `embed_function` in metadata in order to "
                "use the default query function."
            )
        return partial(
            self.query_vector_collection,
            sources=metadata["sources"],
            chunk_strategy=chunk_strategy,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            distance_metric=distance_metric,
            embed_function=embed_function,
        )

    def validate_each_sentence(
        self, value: Any, query_function: Callable, metadata: Dict[str, Any]
    ) -> ValidationResult:
        if nltk is None:
            raise ImportError(
                "`nltk` library is required for `provenance-v0` validator. "
                "Please install it with `poetry add nltk`."
            )
        # Split the value into sentences using nltk sentence tokenizer.
        sentences = nltk.sent_tokenize(value)

        unsupported_sentences = []
        supported_sentences = []
        for sentence in sentences:
            most_similar_chunks = query_function(text=sentence, k=1)
            if most_similar_chunks is None:
                unsupported_sentences.append(sentence)
                continue
            most_similar_chunk = most_similar_chunks[0]
            if most_similar_chunk[1] < self._threshold:
                supported_sentences.append((sentence, most_similar_chunk[0]))
            else:
                unsupported_sentences.append(sentence)

        metadata["unsupported_sentences"] = "- " + "\n- ".join(unsupported_sentences)
        metadata["supported_sentences"] = supported_sentences
        if unsupported_sentences:
            unsupported_sentences = "- " + "\n- ".join(unsupported_sentences)
            return FailResult(
                metadata=metadata,
                error_message=(
                    f"None of the following sentences in your response are supported "
                    "by provided context:"
                    f"\n{metadata['unsupported_sentences']}"
                ),
                fix_value="\n".join(s[0] for s in supported_sentences),
            )
        return PassResult(metadata=metadata)

    def validate_full_text(
        self, value: Any, query_function: Callable, metadata: Dict[str, Any]
    ) -> ValidationResult:
        most_similar_chunks = query_function(text=value, k=1)
        if most_similar_chunks is None:
            metadata["unsupported_text"] = value
            metadata["supported_text_citations"] = {}
            return FailResult(
                metadata=metadata,
                error_message=(
                    "The following text in your response is not supported by the "
                    "supported by the provided context:\n" + value
                ),
            )
        most_similar_chunk = most_similar_chunks[0]
        if most_similar_chunk[1] > self._threshold:
            metadata["unsupported_text"] = value
            metadata["supported_text_citations"] = {}
            return FailResult(
                metadata=metadata,
                error_message=(
                    "The following text in your response is not supported by the "
                    "supported by the provided context:\n" + value
                ),
            )

        metadata["unsupported_text"] = ""
        metadata["supported_text_citations"] = {
            value: most_similar_chunk[0],
        }
        return PassResult(metadata=metadata)

    def validate(self, value: Any, metadata: Dict[str, Any]) -> ValidationResult:
        query_function = self.get_query_function(metadata)

        if self._validation_method == "sentence":
            return self.validate_each_sentence(value, query_function, metadata)
        elif self._validation_method == "full":
            return self.validate_full_text(value, query_function, metadata)
        else:
            raise ValueError("validation_method must be 'sentence' or 'full'.")

    @staticmethod
    def query_vector_collection(
        text: str,
        k: int,
        sources: List[str],
        embed_function: Callable,
        chunk_strategy: str = "sentence",
        chunk_size: int = 5,
        chunk_overlap: int = 2,
        distance_metric: str = "cosine",
    ) -> List[Tuple[str, float]]:
        chunks = [
            get_chunks_from_text(source, chunk_strategy, chunk_size, chunk_overlap)
            for source in sources
        ]
        chunks = list(itertools.chain.from_iterable(chunks))

        # Create embeddings
        source_embeddings = np.array(embed_function(chunks)).squeeze()
        query_embedding = embed_function(text).squeeze()

        # Compute distances
        if distance_metric == "cosine":
            if not _HAS_NUMPY:
                raise ValueError(
                    "You must install numpy in order to use the cosine distance "
                    "metric."
                )

            cos_sim = 1 - (
                np.dot(source_embeddings, query_embedding)
                / (
                    np.linalg.norm(source_embeddings, axis=1)
                    * np.linalg.norm(query_embedding)
                )
            )
            top_indices = np.argsort(cos_sim)[:k]
            top_similarities = [cos_sim[j] for j in top_indices]
            top_chunks = [chunks[j] for j in top_indices]
        else:
            raise ValueError("distance_metric must be 'cosine'.")

        return list(zip(top_chunks, top_similarities))

    def to_prompt(self, with_keywords: bool = True) -> str:
        return ""


@register_validator(name="provenance-v1", data_type="string")
class ProvenanceV1(Validator):
    """Validates that the LLM-generated text is supported by the provided
    contexts.

    This validator uses an LLM callable to evaluate the generated text against the
    provided contexts (LLM-ception).

    In order to use this validator, you must provide either:
    1. a 'query_function' in the metadata. That function should take a string as input
        (the LLM-generated text) and return a list of relevant
    chunks. The list should be sorted in ascending order by the distance between the
        chunk and the LLM-generated text.

    Example using str callable:

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

    Example using a custom llm callable:

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
    ```
    """

    def __init__(
        self,
        validation_method: str = "sentence",
        llm_callable: Union[str, Callable] = "gpt-3.5-turbo",
        top_k: int = 3,
        max_tokens: int = 2,
        on_fail: Optional[Callable] = None,
        **kwargs,
    ):
        """
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
        """
        super().__init__(
            on_fail,
            validation_method=validation_method,
            llm_callable=llm_callable,
            top_k=top_k,
            max_tokens=max_tokens,
            **kwargs,
        )
        if validation_method not in ["sentence", "full"]:
            raise ValueError("validation_method must be 'sentence' or 'full'.")
        self._validation_method = validation_method
        self.set_callable(llm_callable)
        self._top_k = int(top_k)
        self._max_tokens = int(max_tokens)

        self.client = OpenAIClient()

    def set_callable(self, llm_callable: Union[str, Callable]) -> None:
        """Set the LLM callable.

        Args:
            llm_callable: Either the name of the OpenAI model, or a callable that takes
                a prompt and returns a response.
        """
        if isinstance(llm_callable, str):
            if llm_callable not in ["gpt-3.5-turbo", "gpt-4"]:
                raise ValueError(
                    "llm_callable must be one of 'gpt-3.5-turbo' or 'gpt-4'."
                    "If you want to use a custom LLM, please provide a callable."
                    "Check out ProvenanceV1 documentation for an example."
                )

            def openai_callable(prompt: str) -> str:
                response = self.client.create_chat_completion(
                    model=llm_callable,
                    messages=[
                        {"role": "user", "content": prompt},
                    ],
                    max_tokens=self._max_tokens,
                )
                return response.output

            self._llm_callable = openai_callable
        elif isinstance(llm_callable, Callable):
            self._llm_callable = llm_callable
        else:
            raise ValueError(
                "llm_callable must be either a string or a callable that takes a string"
                " and returns a string."
            )

    def get_query_function(self, metadata: Dict[str, Any]) -> Callable:
        # Exact same as ProvenanceV0

        query_fn = metadata.get("query_function", None)
        sources = metadata.get("sources", None)

        # Check that query_fn or sources are provided
        if query_fn is not None:
            if sources is not None:
                warnings.warn(
                    "Both `query_function` and `sources` are provided in metadata. "
                    "`query_function` will be used."
                )
            return query_fn

        if sources is None:
            raise ValueError(
                "You must provide either `query_function` or `sources` in metadata."
            )

        # Check chunking strategy
        chunk_strategy = metadata.get("chunk_strategy", "sentence")
        if chunk_strategy not in ["sentence", "word", "char", "token"]:
            raise ValueError(
                "`chunk_strategy` must be one of 'sentence', 'word', 'char', "
                "or 'token'."
            )
        chunk_size = metadata.get("chunk_size", 5)
        chunk_overlap = metadata.get("chunk_overlap", 2)

        # Check distance metric
        distance_metric = metadata.get("distance_metric", "cosine")
        if distance_metric not in ["cosine", "euclidean"]:
            raise ValueError(
                "`distance_metric` must be one of 'cosine' or 'euclidean'."
            )

        # Check embed model
        embed_function = metadata.get("embed_function", None)
        if embed_function is None:
            raise ValueError(
                "You must provide `embed_function` in metadata in order to "
                "use the default query function."
            )
        return partial(
            self.query_vector_collection,
            sources=metadata["sources"],
            chunk_strategy=chunk_strategy,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            distance_metric=distance_metric,
            embed_function=embed_function,
        )

    @retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(6))
    def call_llm(self, prompt: str) -> str:
        """Call the LLM with the given prompt.

        Expects a function that takes a string and returns a string.

        Args:
            prompt (str): The prompt to send to the LLM.

        Returns:
            response (str): String representing the LLM response.
        """
        return self._llm_callable(prompt)

    def evaluate_with_llm(self, text: str, query_function: Callable) -> bool:
        """Validate that the LLM-generated text is supported by the provided
        contexts.

        Args:
            value (Any): The LLM-generated text.
            query_function (Callable): The query function.

        Returns:
            self_eval: The self-evaluation boolean
        """
        # Get the relevant chunks using the query function
        relevant_chunks = query_function(text=text, k=self._top_k)

        # Create the prompt to ask the LLM
        prompt = PROVENANCE_V1_PROMPT.format(text, "\n".join(relevant_chunks))

        # Get self-evaluation
        self_eval = self.call_llm(prompt)
        self_eval = True if self_eval == "Yes" else False
        return self_eval

    def validate_each_sentence(
        self, value: Any, query_function: Callable, metadata: Dict[str, Any]
    ) -> ValidationResult:
        if nltk is None:
            raise ImportError(
                "`nltk` library is required for `provenance-v0` validator. "
                "Please install it with `poetry add nltk`."
            )
        # Split the value into sentences using nltk sentence tokenizer.
        sentences = nltk.sent_tokenize(value)

        unsupported_sentences = []
        supported_sentences = []
        for sentence in sentences:
            self_eval = self.evaluate_with_llm(sentence, query_function)
            if not self_eval:
                unsupported_sentences.append(sentence)
            else:
                supported_sentences.append(sentence)

        if unsupported_sentences:
            unsupported_sentences = "- " + "\n- ".join(unsupported_sentences)
            return FailResult(
                metadata=metadata,
                error_message=(
                    f"None of the following sentences in your response are supported "
                    "by provided context:"
                    f"\n{unsupported_sentences}"
                ),
                fix_value="\n".join(supported_sentences),
            )
        return PassResult(metadata=metadata)

    def validate_full_text(
        self, value: Any, query_function: Callable, metadata: Dict[str, Any]
    ) -> ValidationResult:
        # Self-evaluate LLM with entire text
        self_eval = self.evaluate_with_llm(value, query_function)
        if not self_eval:
            # if false
            return FailResult(
                metadata=metadata,
                error_message=(
                    "The following text in your response is not supported by the "
                    "supported by the provided context:\n" + value
                ),
            )
        return PassResult(metadata=metadata)

    def validate(self, value: Any, metadata: Dict[str, Any]) -> ValidationResult:
        kwargs = {}
        context_copy = contextvars.copy_context()
        for key, context_var in context_copy.items():
            if key.name == "kwargs" and isinstance(kwargs, dict):
                kwargs = context_var
                break

        api_key = kwargs.get("api_key")
        api_base = kwargs.get("api_base")

        # Set the OpenAI API key
        if os.getenv("OPENAI_API_KEY"):  # Check if set in environment
            self.client.api_key = os.getenv("OPENAI_API_KEY")
        elif api_key:  # Check if set when calling guard() or parse()
            self.client.api_key = api_key

        # Set the OpenAI API base if specified
        if api_base:
            self.client.api_base = api_base

        query_function = self.get_query_function(metadata)
        if self._validation_method == "sentence":
            return self.validate_each_sentence(value, query_function, metadata)
        elif self._validation_method == "full":
            return self.validate_full_text(value, query_function, metadata)
        else:
            raise ValueError("validation_method must be 'sentence' or 'full'.")

    @staticmethod
    def query_vector_collection(
        text: str,
        k: int,
        sources: List[str],
        embed_function: Callable,
        chunk_strategy: str = "sentence",
        chunk_size: int = 5,
        chunk_overlap: int = 2,
        distance_metric: str = "cosine",
    ) -> List[Tuple[str, float]]:
        chunks = [
            get_chunks_from_text(source, chunk_strategy, chunk_size, chunk_overlap)
            for source in sources
        ]
        chunks = list(itertools.chain.from_iterable(chunks))

        # Create embeddings
        source_embeddings = np.array(embed_function(chunks)).squeeze()
        query_embedding = embed_function(text).squeeze()

        # Compute distances
        if distance_metric == "cosine":
            if not _HAS_NUMPY:
                raise ValueError(
                    "You must install numpy in order to use the cosine distance "
                    "metric."
                )

            cos_sim = 1 - (
                np.dot(source_embeddings, query_embedding)
                / (
                    np.linalg.norm(source_embeddings, axis=1)
                    * np.linalg.norm(query_embedding)
                )
            )
            top_indices = np.argsort(cos_sim)[:k]
            top_chunks = [chunks[j] for j in top_indices]
        else:
            raise ValueError("distance_metric must be 'cosine'.")

        return top_chunks
