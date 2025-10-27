from docspec_python import ParserOptions
from docs.pydocs.pydocs_markdown_impl import render_loader
from pydoc_markdown.contrib.loaders.python import PythonLoader
from pydoc_markdown.contrib.processors.filter import FilterProcessor
from docs.pydocs.helpers import write_to_file


export_map = {
    "guardrails/prompt/base_prompt.py": [
        "guardrails.prompt.base_prompt",
        "BasePrompt",
        "__init__",
        "format",
        "substitute_constants",
        "get_prompt_variables",
        "escape",
    ],
    "guardrails/prompt/prompt.py": [
        "guardrails.prompt.prompt",
        "Prompt",
        "format",
    ],
    "guardrails/prompt/instructions.py": [
        "guardrails.prompt.instructions",
        "Instructions",
        "format",
    ],
    "guardrails/llm_providers.py": [
        "guardrails.llm_providers",
        "PromptCallableBase",
        "_invoke_llm",
        "__call__",
    ],
    "guardrails/classes/llm/llm_response.py": [
        "guardrails.classes.llm.llm_response",
        "LLMResponse",
    ],
}


conditionals = []
for k, v in export_map.items():
    conditionals.append(
        f"((name in {v}) if ('{k}' in obj.location.filename) else False)"
    )

export_string = " or ".join(conditionals)

write_to_file(
    str="# Helpers for LLM Interactions\n\n"
    + render_loader(
        PythonLoader(
            modules=[
                "guardrails.prompt.base_prompt",
                "guardrails.prompt.prompt",
                "guardrails.prompt.instructions",
                "guardrails.llm_providers",
                "guardrails.classes.llm.llm_response",
            ],
            parser=ParserOptions(print_function=False),
        ),
        processor=FilterProcessor(
            expression=f"({export_string})",  # noqa
            skip_empty_modules=True,
        ),
    ),
    filename="docs/src/api_reference_markdown/llm_interaction.md",
)
