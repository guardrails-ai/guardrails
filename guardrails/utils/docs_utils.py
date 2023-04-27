import typing as t

from guardrails.prompt import Prompt


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

    def prompt_template_token_length(self, prompt_template: Prompt) -> str:
        """Exclude the tokens used in the prompt template from the text."""
        # TODO(shreya) Make sure that prompt_template.source is correct, and
        # doesn't contain extra metadata.

        prompt_vars = prompt_template.get_prompt_variables()

        tokens = self.tokenizer.encode(
            prompt_template.format(**{var: "" for var in prompt_vars})
        )
        return len(tokens)

    def __call__(self, *args: t.Any, **kwds: t.Any) -> t.Any:
        return self.split(*args, **kwds)


def sentence_split(text: str) -> t.List[str]:
    """Split the text into sentences."""
    try:
        from nltk import sent_tokenize
    except ImportError:
        raise ImportError(
            "nltk is required for sentence splitting. Please install it using "
            "`pip install nltk`"
        )

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
