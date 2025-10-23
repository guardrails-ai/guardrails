from docspec_python import ParserOptions
from docs.docs.pydocs.pydocs_markdown_impl import render_loader
from pydoc_markdown.contrib.loaders.python import PythonLoader
from pydoc_markdown.contrib.processors.filter import FilterProcessor
from pydoc_markdown.contrib.renderers.markdown import MarkdownRenderer
from docs.docs.pydocs.helpers import write_to_file


exports = [
    "guardrails.actions.reask",
    "ReAsk",
    "FieldReAsk",
    "SkeletonReAsk",
    "NonParseableReAsk",
    "guardrails.actions.filter",
    "Filter",
    "apply_filters",
    "guardrails.actions.refrain",
    "Refrain",
    "apply_refrain",
]
export_string = ", ".join([f"'{export}'" for export in exports])

write_to_file(
    str="# Actions\n\n"
    + render_loader(
        PythonLoader(
            modules=[
                "guardrails.actions.reask",
                "guardrails.actions.filter",
                "guardrails.actions.refrain",
            ],
            parser=ParserOptions(print_function=False),
        ),
        processor=FilterProcessor(
            expression=f"name in [{export_string}]",  # noqa
            skip_empty_modules=True,
        ),
        renderer=MarkdownRenderer(
            # Custom
            data_code_block=True,
            data_expression_maxlength=250,
            header_level_by_type={
                "Class": 2,
                "Variable": 2,
            },
            # Default
            render_module_header=False,
            insert_header_anchors=False,
            descriptive_class_title=False,
            signature_in_header=False,
            classdef_code_block=True,
            classdef_with_decorators=True,
            signature_code_block=True,
            signature_with_decorators=True,
            render_typehint_in_data_header=True,
        ),
    ),
    filename="docs/api_reference_markdown/actions.md",
)
