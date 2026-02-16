from docspec_python import ParserOptions
from docs.pydocs.pydocs_markdown_impl import render_loader
from pydoc_markdown.contrib.loaders.python import PythonLoader
from pydoc_markdown.contrib.processors.filter import FilterProcessor
from docs.pydocs.helpers import write_to_file


export_map = {
    "guardrails/guard.py": [
        "Guard",
        "guardrails.guard",
        "guard",
        "__init__",
        "for_rail",
        "for_rail_string",
        "for_pydantic",
        "for_string",
        "configure",
        "use",
        "get_validators",
        "__call__",
        "parse",
        "validate",
        "error_spans_in_output",
        "json_function_calling_tool",
        "to_dict",
        "from_dict",
        "to_runnable",
    ],
    "guardrails/async_guard.py": [
        "AsyncGuard",
        "guardrails.async_guard",
        "async_guard",
        "__init__",
        "for_rail",
        "for_rail_string",
        "for_pydantic",
        "for_string",
        "configure",
        "use",
        "get_validators",
        "__call__",
        "parse",
        "validate",
        "error_spans_in_output",
        "json_function_calling_tool",
        "to_dict",
        "from_dict",
        "to_runnable",
    ],
    "guardrails/classes/validation_outcome.py": [
        "guardrails.classes.validation_outcome",
        "ValidationOutcome",
        "from_guard_history",
    ],
}


conditionals = []
for k, v in export_map.items():
    conditionals.append(
        f"((name in {v}) if ('{k}' in obj.location.filename) else False)"
    )

export_string = " or ".join(conditionals)

write_to_file(
    str="# Guards\n\n"
    + render_loader(
        PythonLoader(
            modules=[
                "guardrails.guard",
                "guardrails.async_guard",
                "guardrails.classes.validation_outcome",
            ],
            parser=ParserOptions(print_function=False),
        ),
        processor=FilterProcessor(
            expression=f"({export_string})",
            skip_empty_modules=True,
        ),
    ),
    filename="docs/src/api_reference_markdown/guards.md",
)
