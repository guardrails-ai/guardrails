import typing as t

from guardrails.prompt import Prompt

try:
    import tiktoken
except ImportError:
    tiktoken = None

try:
    import nltk  # type: ignore
except ImportError:
    nltk = None

if nltk is not None:
    try:
        nltk.data.find("tokenizers/punkt")
    except LookupError:
        nltk.download("punkt")


class TextSplitter:
    """Split the docs into chunks with token boundaries."""

    def __init__(self):
        import tiktoken

        self.tokenizer = tiktoken.get_encoding("gpt2")

    def split(
        self,
        text: str,
        tokens_per_chunk: int = 2048,
        token_overlap: int = 512,
        buffer: int = 128,
        prompt_template: t.Optional[Prompt] = None,
    ) -> t.List[str]:
        # TODO(shreya): Add test to make sure this works correctly.
        """Split the text into chunks with token boundaries."""

        tokens_per_chunk -= buffer

        # If prompt template is provided, create chunks so that the
        # chunk + prompt template is less than tokens_per_chunk.
        if prompt_template:
            tokens_per_chunk -= self.prompt_template_token_length(prompt_template)

        tokens = self.tokenizer.encode(text)
        chunks = []
        for i in range(0, len(tokens), tokens_per_chunk - token_overlap):
            # Note: this is lossy but should be ok.
            chunks.append(self.tokenizer.decode(tokens[i : i + tokens_per_chunk]))
        return chunks

    def prompt_template_token_length(self, prompt_template: Prompt) -> int:
        """Exclude the tokens used in the prompt template from the text."""
        # TODO(shreya) Make sure that prompt_template.source is correct, and
        # doesn't contain extra metadata.

        prompt_vars = prompt_template.get_prompt_variables()

        tokens = self.tokenizer.encode(
            str(prompt_template.format(**{var: "" for var in prompt_vars}))
        )
        return len(tokens)

    def __call__(self, *args: t.Any, **kwds: t.Any) -> t.Any:
        return self.split(*args, **kwds)


def sentence_split(text: str) -> t.List[str]:
    """Split the text into sentences."""
    try:
        from nltk import sent_tokenize  # type: ignore
    except ImportError:
        raise ImportError(
            "nltk is required for sentence splitting. Please install it using "
            "`poetry add nltk`"
        )

    # Download the nltk punkt tokenizer if it's not already downloaded.
    import nltk  # type: ignore

    try:
        nltk.data.find("tokenizers/punkt")
    except LookupError:
        nltk.download("punkt")

    return sent_tokenize(text)


def read_pdf(path) -> str:
    """Reads the pdf at the given path."""
    import pypdfium2 as pdfium

    content = ""
    pdf = pdfium.PdfDocument(path)
    for i in range(len(pdf)):
        page = pdf.get_page(i)
        textpage = page.get_textpage()
        content += textpage.get_text_range()
        content += "\n"
        [g.close() for g in (textpage, page)]
    pdf.close()
    return content.replace("\r", "")


def get_chunks_from_text(
    text: str, chunk_strategy: str, chunk_size: int, chunk_overlap: int
) -> t.List[str]:
    """Get chunks of text from a string.

    Args:
        text: The text to chunk.
        chunk_strategy: The strategy to use for chunking.
        chunk_size: The size of each chunk. If the chunk_strategy is "sentences",
            this is the number of sentences per chunk. If the chunk_strategy is
            "characters", this is the number of characters per chunk, and so on.
        chunk_overlap: The number of characters to overlap between chunks. If the
            chunk_strategy is "sentences", this is the number of sentences to overlap
            between chunks.
    """

    nltk_error = (
        "nltk is required for sentence splitting. Please install it using "
        "`poetry add nltk`"
    )
    tiktoken_error = (
        "tiktoken is required for token splitting. Please install it using "
        "`poetry add tiktoken`"
    )

    if chunk_strategy == "sentence":
        if nltk is None:
            raise ImportError(nltk_error)
        atomic_chunks = nltk.sent_tokenize(text)
    elif chunk_strategy == "word":
        if nltk is None:
            raise ImportError(nltk_error)
        atomic_chunks = nltk.word_tokenize(text)
    elif chunk_strategy == "char":
        atomic_chunks = list(text)
    elif chunk_strategy == "token":
        if tiktoken is None:
            raise ImportError(tiktoken_error)
        # FIXME is this the correct way to use tiktoken?
        atomic_chunks = tiktoken(text)  # type: ignore
    elif chunk_strategy == "full":
        atomic_chunks = [text]
    else:
        raise ValueError(
            "chunk_strategy must be 'sentence', 'word', 'char', or 'token'."
        )

    chunks = []
    for i in range(0, len(atomic_chunks), chunk_size - chunk_overlap):
        chunk = " ".join(atomic_chunks[i : i + chunk_size])
        chunks.append(chunk)
    return chunks
