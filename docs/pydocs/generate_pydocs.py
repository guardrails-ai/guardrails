from docspec_python import ParserOptions
from docs.pydocs.helpers import write_to_file
from docs.pydocs.pydocs_markdown_impl import render_loader
from pydoc_markdown.contrib.loaders.python import PythonLoader

from docs.pydocs.api_reference import actions  # noqa
from docs.pydocs.api_reference import errors  # noqa
from docs.pydocs.api_reference import formatters  # noqa
from docs.pydocs.api_reference import generics_and_base_classes  # noqa
from docs.pydocs.api_reference import guards  # noqa
from docs.pydocs.api_reference import history  # noqa
from docs.pydocs.api_reference import llm_interaction  # noqa
from docs.pydocs.api_reference import types  # noqa
from docs.pydocs.api_reference import validation  # noqa


write_to_file(
    str="# Validators\n\n"
    + render_loader(
        PythonLoader(
            search_path=["validators"], parser=ParserOptions(print_function=False)
        )
    ),
    filename="docs/src/hub/api_reference_markdown/validators.md",
)


# write_to_file(
#     str="# Response Structures\n\n" + render_loader(
#         PythonLoader(
#             modules=['guardrails.validator_base'],
#         ),
#         processor = FilterProcessor(
#             expression="""\
# name in \
# ['guardrails.validator_base', 'ValidationResult', 'PassResult', 'FailResult', 'ValidationError'] \   # noqa
# or obj.parent.name in \
# ['ValidationResult', 'PassResult', 'FailResult', 'ValidationError']\
#             """,
#         ),
#         renderer = MarkdownRenderer(
#             render_module_header=False,
#             insert_header_anchors=False,
#             classdef_code_block=False,
#             descriptive_class_title=False,
#             classdef_with_decorators=False,
#             render_typehint_in_data_header=True,
#             data_code_block=True,
#         )
#     ),
#     filename="docs/src/api_reference_markdown/response_structures.md",
# )


# write_to_file(
#     str="# Schema\n\n" + render_loader(PythonLoader(
#         modules=['guardrails.schema'],
#         parser=ParserOptions(
#             print_function=False
#         )
#     )),
#     filename="docs/src/api_reference_markdown/schema.md",
# )

# write_to_file(
#     str="# Document Store\n\n" + render_loader(
#         loader = PythonLoader(
#             modules=['guardrails.document_store'],
#             parser=ParserOptions(
#             ),
#         ),
#     ),
#     filename="docs/src/api_reference_markdown/document_store.md",
# )

# write_to_file(
#     str=module_to_string(
#         datatypes,
#         ignore_attrs=True,
#         display_string="Data Types",
#         ignore_prefix_list=[
#             "NonScalarType",
#             "ScalarType",
#             "FieldValidation",
#             "DataType",
#         ]
#     ),
#     filename="docs/src/api_reference_markdown/datatypes.md",
# )


# write_to_file(
#     str="# Helper Classes\n\n" + render_loader(
#         PythonLoader(
#             modules=['guardrails.classes.generic.stack'],
#             parser=ParserOptions(
#                 print_function=False
#             ),
#         ),
#         processor = FilterProcessor(
#             # expression="name in ['Stack', 'guardrails.classes.generic.Stack']",
#             skip_empty_modules=True
#         )
#     ),
#     filename="docs/src/api_reference_markdown/helper_classes.md",
# )
