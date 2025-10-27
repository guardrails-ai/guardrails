from docspec_python import ParserOptions
from docs.docs.pydocs.pydocs_markdown_impl import render_loader
from pydoc_markdown.contrib.loaders.python import PythonLoader
from docs.docs.pydocs.helpers import write_to_file


exports = [
    "guardrails.formatters.json_formatter",
    "JsonFormatter",
    "guardrails.formatters.base_formatter",
    "BaseFormatter",
]
export_string = ", ".join([f"'{export}'" for export in exports])

write_to_file(
    str="# Formatters\n\n"
    + render_loader(
        loader=PythonLoader(
            modules=[
                "guardrails.formatters.base_formatter",
                "guardrails.formatters.json_formatter",
            ],
            parser=ParserOptions(print_function=False),
        ),
    ),
    filename="docs/api_reference_markdown/formatters.md",
)
