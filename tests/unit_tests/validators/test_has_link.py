import pytest

from guardrails.validators import FailResult, PassResult
from guardrails.validators.has_link import HasLink


@pytest.mark.parametrize(
    "value, expected_links",
    [
        ("abcdef", []),
        ("See http://example.com for more info.", ["http://example.com"]),
        ("See mystore.com for more info.", []),  # if we match this so will filename.txt
        ("Refer to www2.example.com for more info.", ["www2.example.com"]),
        (
            "Check out https://example.com:8080/test?query=param#anchor.",
            ["https://example.com:8080/test?query=param#anchor"],
        ),
        (
            "Visit our site: https://subdomain.example.co.uk/path.",
            ["https://subdomain.example.co.uk/path"],
        ),
        (
            "Our e-commerce site is https://shop.example-market.com.",
            ["https://shop.example-market.com"],
        ),
        ("Example without TLD: http://localhost/test.", ["http://localhost/test"]),
        (
            "Here's an IP: http://192.168.1.1/setup.html",
            ["http://192.168.1.1/setup.html"],
        ),
        ("With punctuation, visit https://example.com!", ["https://example.com"]),
        (
            "Multiple URLs https://google.com and http://example.com/page.",
            ["https://google.com", "http://example.com/page"],
        ),
        ("Embeddedhttps://embeded.com", []),
        ("Email not a URL: someone@example.com", []),
        ("In brackets (http://example.com)", ["http://example.com"]),
        (
            "With query param http://example.com?search=test",
            ["http://example.com?search=test"],
        ),
        ("FTP ftp://example.com/resource", ["example.com/resource"]),
        (
            "With user info http://user:pass@example.com/test",
            ["http://user:pass@example.com/test"],
        ),
        ("Javascript pseudo URL:javascript:void(0);", []),
        ("Data URI data:text/plain;base64,SGVsbG8sIFdvcmxkIQ==", []),
        (
            "Complex URL https://example.com/path/to/resource?name=value&redir=https://redirect.com#section",  # noqa: E501
            [
                "https://example.com/path/to/resource?name=value&redir=https://redirect.com#section"  # noqa: E501
            ],
        ),
        ("Trailing dot http://example.com.", ["http://example.com"]),
        (
            "URL with encoded space http://example.com/hello%20world",
            ["http://example.com/hello%20world"],
        ),
        (
            "Internationalized URL http://موقع.وزارة-الاتصالات.مصر/",
            ["http://موقع.وزارة-الاتصالات.مصر/"],
        ),
    ],
)
def test_extract_links(value, expected_links):
    assert HasLink._extract_links(value) == expected_links


@pytest.mark.parametrize(
    "value, allowed_domains, is_pass_result, fix_value, disallowed_links",
    [
        ("abcdef", [], True, None, None),
        (
            "See http://example.com for more info.",
            [],
            False,
            "See [REDACTED] for more info.",
            ["http://example.com"],
        ),
        ("See http://example.com for more info.", ["example.com"], True, None, None),
    ],
)
def test_has_link_validator(
    value, allowed_domains, is_pass_result, fix_value, disallowed_links
):
    validator = HasLink(allowed_domains=allowed_domains)
    result = validator.validate(value, {})
    if is_pass_result:
        assert isinstance(result, PassResult)
    else:
        assert isinstance(result, FailResult)
        assert result.fix_value == fix_value

        disallowed_links_bullet_points = "\n".join(
            [f"- {link}" for link in disallowed_links]
        )
        assert (
            result.error_message == f"Value {value} contains disallowed links:\n"
            f"{disallowed_links_bullet_points}"
        )
