

# Create registry of types 
registry = {}


# Create a decorator to register a type
def register_type(name: str):
    def decorator(cls: type):
        registry[name] = cls
        return cls
    return decorator


class DataType:
    def __init__(self):
        self.validators = []


@register_type("string")
class String(DataType):
    pass


@register_type("integer")
class Integer(DataType):
    pass


@register_type("float")
class Float(DataType):
    pass


@register_type("date")
class Date(DataType):
    pass


@register_type("time")
class Time(DataType):
    pass


@register_type("email")
class Email(DataType):
    pass


@register_type("url")
class URL(DataType):
    pass


@register_type("percentage")
class Percentage(DataType):
    pass


@register_type("list")
class List(DataType):
    pass


@register_type("map")
class Map(DataType):
    pass
