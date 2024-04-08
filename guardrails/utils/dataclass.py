import pydantic.version

PYDANTIC_VERSION = pydantic.version.VERSION

if PYDANTIC_VERSION.startswith("1"):

    def dataclass(cls):  # type: ignore
        return cls

else:
    from dataclasses import dataclass  # type: ignore  # noqa
