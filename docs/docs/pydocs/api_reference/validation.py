from docspec_python import ParserOptions
from docs.docs.pydocs.pydocs_markdown_impl import render_loader
from pydoc_markdown.contrib.loaders.python import PythonLoader
from pydoc_markdown.contrib.processors.filter import FilterProcessor
from docs.docs.pydocs.helpers import write_to_file


# exports = [
#     "docs.pydocs.api_reference.validator",
#     "ValidatorReference",
# ]
# export_string = ", ".join([f"'{export}'" for export in exports])


export_map = {
    "guardrails/validator_base.py": [
        "guardrails.validator_base",
        "Validator",
        "__init__",
        "chunking_function",
        "validate",
        "validate_stream",
        "with_metadata",
        "to_runnable",
        "register_validator",
    ],
    "guardrails/classes/validation/validation_result.py": [
        "guardrails.classes.validation.validation_result",
        "ValidationResult",
        "PassResult",
        "FailResult",
        "ErrorSpan",
    ],
    "guardrails/classes/validation/validator_logs.py": [
        "guardrails.classes.validation.validator_logs",
        "ValidatorLogs",
    ],
    "guardrails/classes/validation/validator_reference.py": [
        "guardrails.classes.validation.validator_reference",
        "ValidatorReference",
    ],
}


conditionals = []
for k, v in export_map.items():
    conditionals.append(
        f"((name in {v}) if ('{k}' in obj.location.filename) else False)"
    )

export_string = " or ".join(conditionals)

write_to_file(
    str="# Validation\n\n"
    + render_loader(
        PythonLoader(
            modules=[
                "guardrails.validator_base",
                "guardrails.classes.validation.validation_result",
                "guardrails.classes.validation.validator_logs",
                "guardrails.classes.validation.validator_reference",
            ],
            parser=ParserOptions(print_function=False),
        ),
        processor=FilterProcessor(
            expression=f"({export_string})",
            skip_empty_modules=False,
        ),
    ),
    filename="docs/api_reference_markdown/validator.md",
)
