import pdb
from pydocs_to_md_lib import MarkdownDoc
import pydoc
from guardrails import Rail, Guard, validators, schema, document_store, datatypes
from guardrails.classes import history
from guardrails.classes.validation_outcome import ValidationOutcome
from guardrails.classes.generic import Stack
from pydocs_to_md import class_to_string, module_to_string


def write_pydoc_to_file(obj, file_name):
    with open(file_name, "w") as f:
        doc = pydoc.render_doc(obj, renderer=MarkdownDoc())
        print(doc)
        print("writing doc to file")
        f.write(doc)
        f.close()


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
    str=module_to_string(
        validators,
        ignore_prefix_list=[
            "logger",
            "types_to_validators",
            "validators_registry",
            "EventDetail",
            "Validator",
            "validate",
            "register_validator",
            "PydanticReAsk",
            "Refrain",
            "ValidationResult",
            "PassResult",
            "FailResult",
            "__",
        ],
        display_string="Validators",
    ),
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
    str=module_to_string(
        schema,
        display_string="Schema",
    ),
    filename="docs/api_reference_markdown/schema.md",
)

write_to_file(
    str=module_to_string(
        document_store,
        ignore_prefix_list=["load", "_"],
        display_string="Document Store",
    ),
    filename="docs/api_reference_markdown/document_store.md",
)

write_to_file(
    str=module_to_string(
        datatypes,
        ignore_prefix_list=[
            "get_validators",
            "registry",
            "DataType",
            "register_type",
            "Scalar",
            "set_children",
            "validate",
            "from_str",
            "from_xml",
            "model",
            "validators",
            "to_object_element",
            "_",
        ],
        display_string="Data Types",
    ),
    filename="docs/api_reference_markdown/datatypes.md",
)

write_to_file(
    str=module_to_string(
        history,
        include_list=[
            "Call",
            "CallInputs",
            "Inputs",
            "Iteration",
            "Outputs"
        ],
        display_string="History & Logs",
    ),
    filename="docs/api_reference_markdown/history_and_logs.md",
)

write_to_file(
    str=class_to_string(Stack, ignore_prefix_list=["load", "_"]),
    filename="docs/api_reference_markdown/stack.md",
)
