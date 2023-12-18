from collections import deque
import builtins
import inspect
from pydoc import (
    TextDoc,
    classify_class_attrs,
    classname,
    getdoc,
    _split_list,
    visiblename,
    sort_attributes,
    _is_bound_method,
)


class MarkdownDoc(TextDoc):
    """Formatter class for Markdown documentation."""

    def bold(self, text):
        """Format a string in bold using Markdown syntax."""
        return f"**{text}**"

    def heading(self, level, text):
        """Format a heading with the given level and text."""
        return f"{'#' * level} {text}"

    def indent(self, text, prefix="  "):
        """Indent text by prepending a given prefix to each line."""
        split_doc = text.split("\n")
        for i in range(len(split_doc)):
            split_doc[i] = prefix + split_doc[i]
        return "\n".join(split_doc)

    def section(self, title, contents):
        """Format a section with a given title."""
        clean_contents = self.indent(contents).rstrip()
        return f"\n{self.heading(2, title)}\n{clean_contents}\n\n"

    def formattree(self, tree, modname, parent=None, prefix=" "):
        """Render in Markdown a class tree as returned by inspect.getclasstree()."""
        result = ""
        for entry in tree:
            if type(entry) is tuple:
                c, bases = entry
                result += f"{prefix}{classname(c, modname)}\n"
                # if bases and bases != (parent,):
                # parents = [classname(c, modname) for c in bases]
                # result += f"{self.indent(f'({", ".join(parents)})', prefix='  ')}\n"
            elif type(entry) is list:
                result += self.formattree(entry, modname, c, prefix + "  ")
        return result

    def docroutine(self, object, name=None, mod=None, cl=None):
        """Produce text documentation for a function or method object."""
        realname = object.__name__
        name = name or realname
        note = ''
        skipdocs = 0

        if (inspect.iscoroutinefunction(object) or
                inspect.isasyncgenfunction(object)):
            asyncqualifier = 'async '
        else:
            asyncqualifier = ''

        if name == realname:
            title = f"{self.heading(3, realname)}\n"

        else:
            if cl and inspect.getattr_static(cl, realname, []) is object:
                skipdocs = 1
            title = self.bold(name) + ' = ' + realname
        argspec = None

        if inspect.isroutine(object):
            try:
                signature = inspect.signature(object)
            except (ValueError, TypeError):
                signature = None
            if signature:
                # if there are more than 3 parameters, we'll split up the params so each is on its own line
                argspec = "(\n"
                # iterate through sig param entries and add them to argspec
                paramlist = []
                for param in signature.parameters:
                    paramlist.append(f"{signature.parameters[param]}")
                if len(signature.parameters) > 3:
                    # add two spaces before each param
                    paramlist = [f"  {param}" for param in paramlist]
                    argspec += ",\n".join(paramlist)
                else:
                    argspec += ", ".join(paramlist)

                argspec += "\n)"

                # add return sig if useful
                if signature.return_annotation != inspect.Signature.empty:
                    argspec += f"-> {signature.return_annotation}"

                if realname == '<lambda>':
                    title = self.bold(name) + ' lambda '
                    # XXX lambda's won't usually have func_annotations['return']
                    # since the syntax doesn't support but it is possible.
                    # So removing parentheses isn't truly safe.
                    argspec = argspec[1:-1]  # remove parentheses
        if not argspec:
            argspec = '(...)'
        decl = asyncqualifier + title + f'\n```\n{name}{argspec}\n\n```'

        if skipdocs:
            return decl + '\n'
        else:
            doc = getdoc(object) or ''
            return decl + '\n' + (doc and self.indent(doc).rstrip() + '\n')

    def document(self, object, name=None, mod=None, cl=None):
        """Produce Markdown documentation for a given object."""
        # Implement specific rendering logic for different object types (functions, classes, data)
        # ...
        return super().document(object, name, mod, cl)

    def formatvalue(self, object):
        """Format an argument default value as Markdown text."""
        return f"= {self.repr(object)}"

    def docclass(self, object, name=None, mod=None, *ignored):
        """Produce text documentation for a given class object."""
        realname = object.__name__
        name = name or realname
        bases = object.__bases__

        def makename(c, m=object.__module__):
            return classname(c, m)

        if name == realname:
            title = "# class " + self.bold(realname)
        else:
            title = self.bold(name) + " = class " + realname
        if bases:
            parents = map(makename, bases)
            title = title + "(%s)" % ", ".join(parents)

        contents = []
        push = contents.append

        try:
            signature = inspect.signature(object)
        except (ValueError, TypeError):
            signature = None
        if signature:
            argspec = str(signature)
            if argspec and argspec != "()":
                push(name + argspec + "\n")

        doc = getdoc(object)
        if doc:
            push(doc + "\n")

        # List the mro, if non-trivial.
        mro = deque(inspect.getmro(object))
        if len(mro) > 2:
            push("Method resolution order:")
            for base in mro:
                push("    " + makename(base))
            push("")

        # List the built-in subclasses, if any:
        subclasses = sorted(
            (
                str(cls.__name__)
                for cls in type.__subclasses__(object)
                if not cls.__name__.startswith("_") and cls.__module__ == "builtins"
            ),
            key=str.lower,
        )
        no_of_subclasses = len(subclasses)
        MAX_SUBCLASSES_TO_DISPLAY = 4
        if subclasses:
            push("Built-in subclasses:")
            for subclassname in subclasses[:MAX_SUBCLASSES_TO_DISPLAY]:
                push("    " + subclassname)
            if no_of_subclasses > MAX_SUBCLASSES_TO_DISPLAY:
                push(
                    "    ... and "
                    + str(no_of_subclasses - MAX_SUBCLASSES_TO_DISPLAY)
                    + " other subclasses"
                )
            push("")

        # Cute little class to pump out a horizontal rule between sections.
        class HorizontalRule:
            def __init__(self):
                self.needone = 0

            def maybe(self):
                if self.needone:
                    push("<hr />")
                self.needone = 1

        hr = HorizontalRule()

        def spill(msg, attrs, predicate):
            ok, attrs = _split_list(attrs, predicate)
            if ok:
                hr.maybe()
                push(msg)
                for name, kind, homecls, value in ok:
                    try:
                        value = getattr(object, name)
                    except Exception:
                        # Some descriptors may meet a failure in their __get__.
                        # (bug #1785)
                        push(self.docdata(value, name, mod))
                    else:
                        push(self.document(value, name, mod, object))
            return attrs

        def spilldescriptors(msg, attrs, predicate):
            ok, attrs = _split_list(attrs, predicate)
            if ok:
                hr.maybe()
                push(msg)
                for name, kind, homecls, value in ok:
                    push(self.docdata(value, name, mod))
            return attrs

        def spilldata(msg, attrs, predicate):
            ok, attrs = _split_list(attrs, predicate)
            if ok:
                hr.maybe()
                push(msg)
                for name, kind, homecls, value in ok:
                    doc = getdoc(value)
                    try:
                        obj = getattr(object, name)
                    except AttributeError:
                        obj = homecls.__dict__[name]
                    push(self.docother(obj, name, mod, maxlen=70, doc=doc) + "\n")
            return attrs

        attrs = [
            (name, kind, cls, value)
            for name, kind, cls, value in classify_class_attrs(object)
            if visiblename(name, obj=object)
        ]

        while attrs:
            if mro:
                thisclass = mro.popleft()
            else:
                thisclass = attrs[0][2]
            attrs, inherited = _split_list(attrs, lambda t: t[2] is thisclass)

            if object is not builtins.object and thisclass is builtins.object:
                attrs = inherited
                continue
            elif thisclass is object:
                tag = "defined here"
            else:
                tag = "inherited from %s" % classname(thisclass, object.__module__)

            sort_attributes(attrs, object)

            # Pump out the attrs, segregated by kind.
            attrs = spill("Methods %s:\n" % tag, attrs, lambda t: t[1] == "method")
            attrs = spill(
                "Class methods %s:\n" % tag, attrs, lambda t: t[1] == "class method"
            )
            attrs = spill(
                "Static methods %s:\n" % tag, attrs, lambda t: t[1] == "static method"
            )
            attrs = spilldescriptors(
                "Readonly properties %s:\n" % tag,
                attrs,
                lambda t: t[1] == "readonly property",
            )
            attrs = spilldescriptors(
                "Data descriptors %s:\n" % tag,
                attrs,
                lambda t: t[1] == "data descriptor",
            )
            attrs = spilldata(
                "Data and other attributes %s:\n" % tag, attrs, lambda t: t[1] == "data"
            )

            assert attrs == []
            attrs = inherited

        contents = "\n".join(contents)
        if not contents:
            return title + "\n"
        return title + "\n" + self.indent(contents.rstrip(), "") + "\n"