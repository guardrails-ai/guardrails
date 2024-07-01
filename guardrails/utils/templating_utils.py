import collections
from string import Template
from typing import List


def get_template_variables(template: str) -> List[str]:
    if hasattr(Template, "get_identifiers"):
        return Template(template).get_identifiers()  # type: ignore
    else:
        d = collections.defaultdict(str)
        Template(template).safe_substitute(d)
        return list(d.keys())
