# Read a pdf stored at a given path
import typing as t

import pypdfium2 as pdfium
import tiktoken
from jinja2 import Template
from openai import Completion

from pydantic import BaseModel, validator


class TextSplitter:
    """Split the docs into chunks with token boundaries."""
    def __init__(self):
        self.tokenizer = tiktoken.get_encoding("gpt2")

    def split(
        self,
        text: str,
        tokens_per_chunk: int = 2048,
        token_overlap: int = 512,
        buffer: int = 128,
        prompt_template: t.Optional[Template] = None,
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
            chunks.append(self.tokenizer.decode(tokens[i:i + tokens_per_chunk]))
        return chunks

    def prompt_template_token_length(self, prompt_template: Template) -> str:
        """Exclude the tokens used in the prompt template from the text."""
        # TODO(shreya) Make sure that prompt_template.source is correct, and
        # doesn't contain extra metadata.
        tokens = self.tokenizer.encode(prompt_template.render({'document': ''}))
        return len(tokens)

    def __call__(self, *args: t.Any, **kwds: t.Any) -> t.Any:
        return self.split(*args, **kwds)


class SchemaExtractor:
    def __init__(self):
        self.template = Template(open('extract_data_prompt.html').read())

    def get_llm_output(self, document):
        prompt = self.template.render(document=document)

        # Number of tokens in the prompt.
        tokenizer = tiktoken.get_encoding("gpt2")
        print(f"prompt length: {len(tokenizer.encode(prompt))}")

        response = Completion.create(model="text-davinci-003", prompt=prompt, temperature=0, max_tokens=2048)
        return response


class Extractor:
    """Given a pdf, extracts the info from it."""

    def __init__(self):
        self.splitter = TextSplitter()
        self.schema_extractor = SchemaExtractor()

    def _read_pdf(self, path):
        """Reads the pdf at the given path."""
        content = ""
        pdf = pdfium.PdfDocument(path)
        for i in range(len(pdf)):
            page = pdf.get_page(i)
            textpage = page.get_textpage()
            content += textpage.get_text()
            content += "\n"
            [g.close() for g in (textpage, page)]
        pdf.close()
        return content

    def extract_from_chunk(self, chunk: str) -> t.List[t.Dict[str, t.Any]]:
        """Extracts the info from the given chunk."""
        llm_output = self.schema_extractor.get_llm_output(document=chunk)
        extracted_info = llm_output['choices'][0]['text']
        print(f"extracted_info: {extracted_info}")
        return extracted_info

    # def combine_chunks(self, chunks: t.List[str]) -> str:
    #     """Combine the info extracted from each chunk into a single string."""

    def extract(self, path: str) -> t.List[t.Dict[str, t.Any]]:
        """Extracts the info from the pdf at the given path."""
        content = self._read_pdf(path)
        content_chunks = self.splitter(content, prompt_template=self.schema_extractor.template)

        for chunk in content_chunks:
            self.extract_from_chunk(chunk)
            break

        return


if __name__ == '__main__':

    extractor = Extractor()
    extractor.extract("chase_zelle.pdf")
