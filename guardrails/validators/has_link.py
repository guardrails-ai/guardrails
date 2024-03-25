import re
from typing import Any, Dict, List, Optional

from guardrails.validator_base import (
    FailResult,
    PassResult,
    ValidationResult,
    Validator,
    register_validator,
)


@register_validator(name="has-link", data_type="string")
class HasLink(Validator):
    """Validates if a string contains a link that is not in the allowed
    domains.

    **Key Properties**

    | Property                      | Description                         |
    | ----------------------------- | ----------------------------------- |
    | Name for `format` attribute   | `has-link`                          |
    | Supported data types          | `string`                            |
    | Programmatic fix              | Remove links not in allowed domains |

    Args:
        allowed_domains (List[str], optional): A list of allowed domains,
            matched as substring of link. Defaults to empty list.
        replace_value (str, optional): The value to replace a disallowed link with
            in the fix value. Defaults to [REDACTED].
    """

    def __init__(
        self,
        allowed_domains: Optional[List[str]] = None,
        replace_value: str = "[REDACTED]",
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        if allowed_domains is None:
            allowed_domains = []
        self.allowed_domains = allowed_domains
        self.replace_value = replace_value

    @staticmethod
    def _extract_links(value: str) -> List[str]:
        """Extracts links from a string.

        Args:
            value (str): The string to extract links from.

        Returns:
            List[str]: A list of links.
        """
        # regex to extract links from the string
        # see: https://daringfireball.net/2010/07/improved_regex_for_matching_urls
        link_regex = r"""(?i)\b((?:https?://|www\d{0,3}[.]|[a-z0-9.\-]+[.][a-z]{2,4}/)(?:[^\s()<>]+|\(([^\s()<>]+|(\([^\s()<>]+\)))*\))+(?:\(([^\s()<>]+|(\([^\s()<>]+\)))*\)|[^\s`!()\[\]{};:'".,<>?«»“”‘’]))"""  # noqa: E501
        return [match[0] for match in re.findall(link_regex, value)]

    def validate(self, value: Any, metadata: Dict) -> ValidationResult:
        # extract links from the string
        links = self._extract_links(value)

        # if no links are found, return a pass result
        if not links:
            return PassResult()

        # create fix value by replacing links not in allowed domains
        fix_value = value
        is_fail = False
        disallowed_links = []
        for link in links:
            if not any(domain in link for domain in self.allowed_domains):
                is_fail = True
                disallowed_links.append(link)
                fix_value = fix_value.replace(link, self.replace_value)

        # if allowed domains are not specified, return a fail result
        if is_fail:
            disallowed_links_bullet_points = "\n".join(
                [f"- {link}" for link in disallowed_links]
            )
            return FailResult(
                error_message=f"Value {value} contains disallowed links:\n"
                f"{disallowed_links_bullet_points}",
                fix_value=fix_value,
            )

        return PassResult()
