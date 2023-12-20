from docspec_python import ParserOptions
from pydoc_markdown.interfaces import Context
from pydoc_markdown.contrib.loaders.python import PythonLoader
from pydoc_markdown.contrib.renderers.markdown import MarkdownRenderer, MarkdownReferenceResolver
from pydoc_markdown.contrib.processors.filter import FilterProcessor
from pydoc_markdown.contrib.processors.google import GoogleProcessor


def load_validators():
    context = Context(directory='guardrails')
    loader = PythonLoader(
        search_path=['validators'],
        parser=ParserOptions(
            print_function=False
        )
    )

    renderer = MarkdownRenderer(
        render_module_header=False,
        insert_header_anchors=False,
        descriptive_class_title=False,
        signature_in_header=True,
        classdef_code_block=False,
        classdef_with_decorators=False,
    )

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


def load_document_store():
    context = Context(directory='guardrails')
    loader = PythonLoader(
        modules=['guardrails.document_store'],
        parser=ParserOptions(
            print_function=False
        )
    )

    renderer = MarkdownRenderer(
        render_module_header=False,
        insert_header_anchors=False,
        descriptive_class_title=False,
        signature_in_header=True,
        classdef_code_block=False,
        classdef_with_decorators=False,
    )

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

def render_loader(loader):
    context = Context(directory='guardrails')

    renderer = MarkdownRenderer(
        render_module_header=False,
        insert_header_anchors=False,
        descriptive_class_title=False,
        signature_in_header=True,
        classdef_code_block=False,
        classdef_with_decorators=False,
    )

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