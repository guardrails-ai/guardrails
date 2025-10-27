from docspec_python import ParserOptions
from docs.docs.pydocs.pydocs_markdown_impl import render_loader
from pydoc_markdown.contrib.loaders.python import PythonLoader
from docs.docs.pydocs.helpers import write_to_file


write_to_file(
    str="# History and Logs\n\n"
    + render_loader(
        loader=PythonLoader(
            modules=[
                "guardrails.classes.history.call",
                "guardrails.classes.history.iteration",
                "guardrails.classes.history.inputs",
                "guardrails.classes.history.outputs",
                "guardrails.classes.history.call_inputs",
            ],
            parser=ParserOptions(print_function=False),
        ),
    ),
    filename="docs/api_reference_markdown/history_and_logs.md",
)
