from docspec_python import ParserOptions
from docs.pydocs.pydocs_markdown_impl import load_validators, render_loader
from pydoc_markdown.contrib.loaders.python import PythonLoader
from guardrails import Rail, Guard, validators, datatypes
from guardrails.classes.validation_outcome import ValidationOutcome
from guardrails.classes import generic
from pydocs_to_md import class_to_string, module_to_string



def write_to_file(str, filename):
    with open(filename, "w") as f:
        f.write(str)
        f.close()


write_to_file(
    str=class_to_string(Rail, ignore_prefix_list=["load", "_"]),
    filename="docs/api_reference_markdown/rail.md",
)


write_to_file(
    str=class_to_string(
        Guard,
        include_list=[
            "from_rail",
            "from_rail_string",
            "from_pydantic",
            "from_string",
            "configure",
            "__call__",
            "parse",
            "state",
        ],
    ),
    filename="docs/api_reference_markdown/guard.md",
)


write_to_file(
    str=f"# Validators\n\n{load_validators()}",
    filename="docs/api_reference_markdown/validators.md",
)

write_to_file(
    str=class_to_string(ValidationOutcome, ignore_prefix_list=["load", "_"]),
    filename="docs/api_reference_markdown/validation_outcome.md",
)

write_to_file(
    str=module_to_string(
        validators,
        include_list=[
            "ValidationResult",
            "PassResult",
            "FailResult",
            "ValidationError",
        ],
        display_string="Response Structures",
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

# write_to_file(
#     str=f"# Document Store\n\n{load_document_store()}",
#     filename="docs/api_reference_markdown/document_store.md",
# )


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
    str="# History and Logs\n\n" + render_loader(PythonLoader(
        packages=['classes.history'],
        parser=ParserOptions(
            print_function=False
        )
    )),
    filename="docs/api_reference_markdown/history_and_logs.md",
)

write_to_file(
    str=module_to_string(
        generic,
        ignore_prefix_list=["load", "_"],
        display_string="Helper Classes",
        include_list=["Stack"],
    ),
    filename="docs/api_reference_markdown/stack.md",
)
