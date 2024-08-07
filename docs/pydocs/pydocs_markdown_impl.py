from pydoc_markdown.interfaces import Context
from pydoc_markdown.contrib.renderers.markdown import (
    MarkdownRenderer,
    MarkdownReferenceResolver,
)
from pydoc_markdown.contrib.processors.filter import FilterProcessor
from pydoc_markdown.contrib.processors.google import GoogleProcessor
# from pydoc_markdown.contrib.processors.crossref import CrossrefProcessor


def render_loader(loader, processor=None, renderer=None, context=None):
    if not context:
        context = Context(directory="guardrails")

    if not renderer:
        new_renderer = MarkdownRenderer(
            render_module_header=False,
            insert_header_anchors=False,
            descriptive_class_title=False,
            signature_in_header=False,
            classdef_code_block=True,
            classdef_with_decorators=True,
            signature_code_block=True,
            signature_with_decorators=True,
            render_typehint_in_data_header=True,
        )
        renderer = new_renderer

    if not processor:
        processor = FilterProcessor(
            skip_empty_modules=True,
        )

    google_processor = GoogleProcessor()

    # TODO: Add this for api reference
    # cross_ref_processor = CrossrefProcessor()

    loader.init(context)
    renderer.init(context)
    processor.init(context)

    modules = list(loader.load())

    processor.process(modules=modules, resolver=MarkdownReferenceResolver())
    google_processor.process(modules=modules, resolver=MarkdownReferenceResolver())
    return renderer.render_to_string(modules)
