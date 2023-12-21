import os
from pydoc_markdown.interfaces import Context
from docspec_python import ParserOptions
from docs.pydocs.pydocs_markdown_impl import render_loader
from pydoc_markdown.contrib.loaders.python import PythonLoader
from guardrails import Rail, Guard, validators, datatypes
from guardrails.classes.validation_outcome import ValidationOutcome
from pydoc_markdown.contrib.renderers.markdown import MarkdownRenderer
from guardrails.classes import generic
from pydocs_to_md import class_to_string, module_to_string
from pydoc_markdown.contrib.processors.filter import FilterProcessor



def write_to_file(str, filename):
    # if the directory where the filename does not exist, create it
    if not os.path.exists(os.path.dirname(filename)):
        os.makedirs(os.path.dirname(filename))
    with open(filename, "w") as f:
        f.write(str)
        f.close()


write_to_file(
    # str=class_to_string(Rail, ignore_prefix_list=["load", "_"]),
    str="# Rail\n\n" + render_loader(
        PythonLoader(
            modules=['guardrails.Rail'],
            parser=ParserOptions(
                print_function=False,
            ),
        ),
        processor = FilterProcessor(
            expression="not name.startswith('_') and not name.startswith('load') and default()",
            documented_only=True,
            
        )
    ),
    filename="docs/api_reference_markdown/rail.md",
)


write_to_file(
    str="# Guard\n\n" + render_loader(
        PythonLoader(
            modules=['guardrails.Guard'],
            parser=ParserOptions(
                print_function=False
            ),
        ),
        processor = FilterProcessor(
            expression="name in ['Guard', 'guardrails.Guard', 'from_rail', 'from_rail_string', 'from_pydantic', 'from_string', 'configure', '__call__', 'parse', 'state']",
            skip_empty_modules=True
        )
    ),
    filename="docs/api_reference_markdown/guard.md",
)


write_to_file(
    str="# Validators\n\n" + render_loader(
        PythonLoader(
            search_path=['validators'],
            parser=ParserOptions(
                print_function=False
            )
        )
    ),
    filename="docs/api_reference_markdown/validators.md",
)

write_to_file(
    str=class_to_string(ValidationOutcome, ignore_prefix_list=["load", "_"]),
    filename="docs/api_reference_markdown/validation_outcome.md",
)

write_to_file(
    str="# Response Structures\n\n" + render_loader(
        PythonLoader(
            modules=['guardrails.validator_base'],
        ),
        processor = FilterProcessor(
            expression="""\
name in \
['guardrails.validator_base', 'ValidationResult', 'PassResult', 'FailResult', 'ValidationError'] \
or obj.parent.name in \
['ValidationResult', 'PassResult', 'FailResult', 'ValidationError']\
            """,
        ),
        renderer = MarkdownRenderer(
            render_module_header=False,
            insert_header_anchors=False,
            classdef_code_block=False,
            descriptive_class_title=False,
            classdef_with_decorators=False,
            render_typehint_in_data_header=True,
            data_code_block=True,
        )
    ),
    filename="docs/api_reference_markdown/response_structures.md",
)



write_to_file(
    str="# Schema\n\n" + render_loader(PythonLoader(
        modules=['guardrails.schema'],
        parser=ParserOptions(
            print_function=False
        )
    )),
    filename="docs/api_reference_markdown/schema.md",
)

write_to_file(
    str="# Document Store\n\n" + render_loader(
        loader = PythonLoader(
            modules=['guardrails.document_store'],
            parser=ParserOptions(
            ),
        ),
    ),
    filename="docs/api_reference_markdown/document_store.md",
)

write_to_file(
    str=module_to_string(
        datatypes,
        ignore_attrs=True,
        display_string="Data Types",
        ignore_prefix_list=[
            "NonScalarType",
            "ScalarType",
            "FieldValidation",
            "DataType",
        ]
    ),
    filename="docs/api_reference_markdown/datatypes.md",
)

write_to_file(
    str="# History and Logs\n\n" + render_loader(
        PythonLoader(
            packages=['classes.history'],
            parser=ParserOptions(
                print_function=False
            ),
        ),
        renderer = MarkdownRenderer(
            render_module_header=True,
            insert_header_anchors=False,
            descriptive_class_title=True,
            signature_in_header=True,
            classdef_code_block=False,
            classdef_with_decorators=False,
        ),
    ),
    filename="docs/api_reference_markdown/history_and_logs.md",
)

write_to_file(
    # str=module_to_string(
    #     generic,
    #     ignore_prefix_list=["load", "_"],
    #     display_string="Helper Classes",
    #     include_list=["Stack"],
    # ),
    str="# Stack\n\n" + render_loader(
        PythonLoader(
            modules=['guardrails.classes.generic.Stack'],
            parser=ParserOptions(
                print_function=False
            ),
        ),
        processor = FilterProcessor(
            # expression="name in ['Stack', 'guardrails.classes.generic.Stack']",
            skip_empty_modules=True
        )
    ),
    filename="docs/api_reference_markdown/stack.md",
)
