from pydoc_markdown.interfaces import Context
from pydoc_markdown.contrib.renderers.markdown import MarkdownRenderer, MarkdownReferenceResolver
from pydoc_markdown.contrib.processors.filter import FilterProcessor
from pydoc_markdown.contrib.processors.google import GoogleProcessor

def render_loader(loader, processor = None, renderer = None, context = None):
    if not context:
        context = Context(directory='guardrails')

    if not renderer:
        renderer = MarkdownRenderer(
            render_module_header=False,
            insert_header_anchors=False,
            descriptive_class_title=False,
            signature_in_header=True,
            classdef_code_block=False,
            classdef_with_decorators=False,
        )

    if not processor:
        processor = FilterProcessor(
            skip_empty_modules=True,
        )

    google_processor = GoogleProcessor()

    loader.init(context)
    renderer.init(context)
    processor.init(context)


    modules = list(loader.load())

    processor.process(modules=modules, resolver=MarkdownReferenceResolver())
    google_processor.process(modules=modules, resolver=MarkdownReferenceResolver())
    return renderer.render_to_string(modules)