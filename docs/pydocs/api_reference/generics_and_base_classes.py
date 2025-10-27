from docspec_python import ParserOptions
from docs.pydocs.pydocs_markdown_impl import render_loader
from pydoc_markdown.contrib.loaders.python import PythonLoader
from docs.pydocs.helpers import write_to_file


exports = [
    "guardrails.classes.generic.arbitrary_model",
    "ArbitraryModel",
    "guardrails.classes.generic.stack",
    "Stack",
]
export_string = ", ".join([f"'{export}'" for export in exports])

write_to_file(
    str="# Generics And Base Classes\n\n"
    + render_loader(
        loader=PythonLoader(
            modules=[
                "guardrails.classes.generic.arbitrary_model",
                "guardrails.classes.generic.stack",
            ],
            parser=ParserOptions(print_function=False),
        ),
    ),
    filename="docs/src/api_reference_markdown/generics_and_base_classes.md",
)
