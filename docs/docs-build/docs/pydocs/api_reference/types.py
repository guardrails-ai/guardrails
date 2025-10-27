from docspec_python import ParserOptions
from docs.docs.pydocs.pydocs_markdown_impl import render_loader
from pydoc_markdown.contrib.loaders.python import PythonLoader
from pydoc_markdown.contrib.processors.filter import FilterProcessor
from pydoc_markdown.contrib.renderers.markdown import MarkdownRenderer
from docs.docs.pydocs.helpers import write_to_file


exports = [
    "guardrails.types.on_fail",
    "guardrails.types.rail",
    "OnFailAction",
    "RailTypes",
    "guardrails.types.inputs",
    "guardrails.types.pydantic",
    "guardrails.types.validator",
    "MessageHistory",
    "ModelOrListOfModels",
    "ModelOrListOrDict",
    "ModelOrModelUnion",
    "PydanticValidatorTuple",
    "PydanticValidatorSpec",
    "UseValidatorSpec",
    "UseManyValidatorTuple",
    "UseManyValidatorSpec",
    "ValidatorMap",
]
export_string = ", ".join([f"'{export}'" for export in exports])

write_to_file(
    str="# Types\n\n"
    + render_loader(
        PythonLoader(
            modules=[
                "guardrails.types.on_fail",
                "guardrails.types.primitives",
                "guardrails.types.rail",
                "guardrails.types.inputs",
                "guardrails.types.pydantic",
                "guardrails.types.validator",
            ],
            parser=ParserOptions(
                print_function=False, treat_singleline_comment_blocks_as_docstrings=True
            ),
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
                "Method": 2,
                "Function": 2,
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
    filename="docs/api_reference_markdown/types.md",
)
