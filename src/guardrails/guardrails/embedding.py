from abc import ABC, abstractmethod
from functools import cached_property
from itertools import islice
from typing import Callable, List, Optional

from guardrails.utils.openai_utils import OpenAIClient


class EmbeddingBase(ABC):
    """Base class for embedding models."""

    def __init__(
        self,
        model: Optional[str] = None,
        encoding_name: Optional[str] = None,
        max_tokens: Optional[int] = None,
    ):
        try:
            import numpy  # noqa: F401
        except ImportError:
            raise ImportError(
                f"`numpy` is required for `{self.__class__.__name__}` class."
                "Please install it with `poetry add numpy`."
            )

        self._model = model
        self._encoding_name = encoding_name
        self._max_tokens = max_tokens

    @abstractmethod
    def embed(self, texts: List[str]) -> List[List[float]]:
        """Embeds a list of texts and returns a list of vectors of floats."""
        ...

    @abstractmethod
    def embed_query(self, query: str) -> List[float]:
        """Embeds a single query and returns a vector of floats."""
        ...

    def _len_safe_get_embedding(
        self, text, embedder: Callable[[str], List[float]], average=True
    ) -> List[float]:
        """Gets the embedding for a text, but splits it into chunks if it is
        too long.

        Args:
            text: Text to embed.
            embedder: Embedding function to use.
            average: Whether to average the embeddings of the chunks.
        Returns:
            List[float] Embedding of the text.
        """
        try:
            import numpy as np
        except ImportError:
            raise ImportError(
                f"`numpy` is required for `{self.__class__.__name__}` class."
                "Please install it with `poetry add numpy`."
            )

        chunk_embeddings_list = []
        chunk_lens = []

        for chunk in EmbeddingBase._chunked_tokens(
            text=text, encoding_name=self._encoding_name, chunk_length=self._max_tokens
        ):
            chunk_embeddings_list.append(embedder(chunk))
            chunk_lens.append(len(chunk))

        if average:
            chunk_embeddings = np.average(
                chunk_embeddings_list, axis=0, weights=chunk_lens
            )
            chunk_embeddings = chunk_embeddings / np.linalg.norm(
                chunk_embeddings
            )  # normalizes length to 1
        else:
            chunk_embeddings = np.array(chunk_embeddings_list)
        return chunk_embeddings.flatten().tolist()

    @staticmethod
    def _chunked_tokens(text, encoding_name, chunk_length):
        """Calculates the number of tokens and chunks them into chunks of
        tokens."""
        import tiktoken

        encoding = tiktoken.get_encoding(encoding_name)
        tokens = encoding.encode(text)
        chunks_iterator = EmbeddingBase._batched(iterable=tokens, n=chunk_length)
        # Detokenize the chunks
        for chunk in chunks_iterator:
            yield encoding.decode(chunk)

    @staticmethod
    def _batched(iterable, n):
        """Batch data into tuples of length n.

        The last batch may be shorter.
        """
        # batched('ABCDEFG', 3) --> ABC DEF G
        if n < 1:
            raise ValueError("n must be at least one")
        it = iter(iterable)
        while batch := list(islice(it, n)):
            yield batch

    @property
    def output_dim(self) -> int:
        """Returns the dimension of the output of the model."""
        raise NotImplementedError


class OpenAIEmbedding(EmbeddingBase):
    def __init__(
        self,
        model: str = "text-embedding-ada-002",
        encoding_name: str = "cl100k_base",
        max_tokens: int = 8191,
        api_key: Optional[str] = None,
        api_base: Optional[str] = None,
    ):
        super().__init__(model, encoding_name, max_tokens)
        self._model = model
        self.api_key = api_key
        self.api_base = api_base

    def embed(self, texts: List[str]) -> List[List[float]]:
        embeddings = []
        for text in texts:
            embeddings.append(super()._len_safe_get_embedding(text, self.embed_query))

        return embeddings

    def embed_query(self, query: str) -> List[float]:
        resp = self._get_embedding([query])
        return resp[0]

    def _get_embedding(self, texts: List[str]) -> List[List[float]]:
        client = OpenAIClient(
            api_key=self.api_key,
            api_base=self.api_base,
        )
        return client.create_embedding(
            model=self._model,
            input=texts,
        )

    @property
    def output_dim(self) -> int:
        if self._model is None:
            raise ValueError("Model not set")
        if self._model == "text-embedding-ada-002":
            return 1536
        elif "ada" in self._model:
            return 1024
        elif "babbage" in self._model:
            return 2048
        elif "curie" in self._model:
            return 4096
        elif "davinci" in self._model:
            return 12288
        else:
            raise ValueError("Unknown model")


class ManifestEmbedding(EmbeddingBase):
    def __init__(
        self,
        client_name: str = "openai",
        client_connection: Optional[str] = None,
        cache_name: Optional[str] = None,
        cache_connection: Optional[str] = None,
        engine: Optional[str] = "text-embedding-ada-002",
        encoding_name: Optional[str] = "cl100k_base",
        max_tokens: Optional[int] = 8191,
    ):
        try:
            from manifest import Manifest  # type: ignore
        except ImportError:
            raise ImportError(
                "The `manifest` package is not installed. "
                "Install with `poetry add manifest-ml`"
            )
        super().__init__(engine, encoding_name, max_tokens)
        self._client_name = client_name
        self._client_connection = client_connection
        self._cache_name = cache_name
        self._cache_connection = cache_connection

        manifest_args = {
            "client_name": client_name,
            "client_connection": client_connection,
            "cache_name": cache_name,
            "cache_connection": cache_connection,
            "engine": engine,
        }
        manifest_args = {k: v for k, v in manifest_args.items() if v is not None}
        self._manifest = Manifest(**manifest_args)

    def embed(self, texts: List[str]) -> List[List[float]]:
        embeddings = []
        for text in texts:
            embeddings.append(super()._len_safe_get_embedding(text, self.embed_query))

        return embeddings

    def embed_query(self, query: str) -> List[float]:
        resp = self._get_embedding([query])
        return resp[0]

    def _get_embedding(self, texts: List[str]) -> List[List[float]]:
        embeddings = self._manifest.run(texts)
        return embeddings  # type: ignore

    @cached_property
    def output_dim(self) -> int:
        embedding = self._get_embedding(["test"])
        return len(embedding[0])
