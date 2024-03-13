import pydantic.version

PYDANTIC_VERSION = pydantic.version.VERSION

if PYDANTIC_VERSION.startswith("1"):
    from pydantic.dataclasses import dataclass  # type: ignore
else:
    from dataclasses import dataclass  # type: ignore  # noqa
