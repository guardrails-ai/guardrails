import inspect


def check_if_prefixed(text, prefix_list):
    return any(text.startswith(prefix) for prefix in prefix_list)


def module_to_string(
    module,
    display_string,
    ignore_prefix_list=[],
    include_list=[],
    indents=1,
    visited=set(),
    ignore_attrs=False,
):
    if module in visited:
        return ""
    visited.add(module)

    module_str = f"{'#' * indents} {display_string}\n"
    module_docs = inspect.getdoc(module)
    if module_docs:
        module_str += f"\n{module_docs}\n\n"

    try:
        for name, obj in inspect.getmembers(module):
            if ignore_prefix_list != [] and check_if_prefixed(name, ignore_prefix_list):
                continue
            if include_list != [] and name not in include_list:
                continue

            if inspect.isclass(obj) or inspect.ismodule(obj):
                # ignore any class that does not belong to this module
                #   or do not have a module
                # ignore if no module
                if not hasattr(obj, "__module__") or not obj.__module__:
                    continue

                if not obj.__module__.startswith(module.__name__) and obj.__module__ != "builtins":
                    continue
                unwrapped = module_to_string(
                    obj,
                    ignore_prefix_list=ignore_prefix_list,
                    ignore_attrs=ignore_attrs,
                    indents=2,
                    # include_list=include_list,
                    display_string=obj.__name__,
                    visited=visited,
                )
                module_str += f"{unwrapped}\n"
            elif not ignore_attrs:
                module_str += f"{attr_to_string(name, obj)}\n"
    except Exception as e:
        print(f"failed to get members of {module.__name__}")
        print(e)
        pass

    return module_str


def class_to_string(cls, ignore_prefix_list=[], include_list=[], indents=1, display_string=None):
    if display_string is None:
        display_string = cls.__name__
    return module_to_string(
        cls,
        display_string,
        ignore_prefix_list,
        include_list,
        indents=indents,
    )


def attr_to_string(name, obj):
    docstring = inspect.getdoc(obj)
    obj_type = clean_obj_type_str(f"{type(obj)}")
    attr_str = f"### {name} `{obj_type}`\n"

    if inspect.isroutine(obj):
        attr_str += routine_to_string(name, obj)

    if docstring:
        attr_str += f"""\n{docstring}\n"""
        # print(f"Properties of {obj.__class__.__name__}:")
        # for prop in properties:
        #     print(f"  - {prop}")

    return attr_str


def routine_to_string(name, obj):
    argspec = ""
    try:
        signature = inspect.signature(obj)
    except (ValueError, TypeError):
        signature = None
    if signature:
        argspec = "(\n"
        paramlist = []
        for param in signature.parameters:
            paramlist.append(f"  {signature.parameters[param]}")

        argspec += ",\n".join(paramlist)

        argspec += "\n)"

        # add return sig if useful
        if signature.return_annotation != inspect.Signature.empty:
            argspec += f" -> {signature.return_annotation}"

    return f"\n```\n{name}{argspec}\n```\n"


# precondition: obj is a function
# def print_method_headers(code):
# pydoc.
# tree = ast.parse(code)
# for node in ast.walk(tree):
#     if isinstance(node, ast.FunctionDef):
#         args = ", ".join(arg.arg for arg in node.args.args)
#         print(f"{node.name}({args})")


def clean_obj_type_str(obj_type_str):
    return obj_type_str.replace("<class '", "class").replace("'>", "")
