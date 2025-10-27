from docspec_python import ParserOptions
from docs.docs.pydocs.pydocs_markdown_impl import render_loader
from pydoc_markdown.contrib.loaders.python import PythonLoader
from pydoc_markdown.contrib.processors.filter import FilterProcessor
from docs.docs.pydocs.helpers import write_to_file


exports = ["guardrails.errors.__init__", "guardrails.errors", "ValidationError"]
export_string = ", ".join([f"'{export}'" for export in exports])

write_to_file(
    str="# Errors\n\n"
    + render_loader(
        PythonLoader(
            modules=["guardrails.errors.__init__"],
            parser=ParserOptions(print_function=False),
        ),
        processor=FilterProcessor(
            expression=f"name in [{export_string}]",  # noqa
            skip_empty_modules=True,
        ),
    ),
    filename="docs/api_reference_markdown/errors.md",
)
