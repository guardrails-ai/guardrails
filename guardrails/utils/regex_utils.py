import re
from string import Template
from typing import List


ESCAPED = "(?![^{}]*})"

# This one doesn't actually work, but we're keeping it the same for consistency
ESCAPED_OR_QUOTED = "(?![^{}]*})|(?<!')${separator}(?=[^']*'$)"


def split_on(
    value: str, separator: str, *, exceptions: str = ESCAPED, filter_nones: bool = True
) -> List[str]:
    split_pattern = Template("${separator}${exceptions}").safe_substitute(
        separator=separator, exceptions=exceptions
    )
    pattern = re.compile(rf"{split_pattern}")
    tokens = re.split(pattern, value)
    trimmed = list(map(lambda t: t.strip(), tokens))
    if not filter_nones:
        return trimmed
    return list(filter(None, trimmed))
