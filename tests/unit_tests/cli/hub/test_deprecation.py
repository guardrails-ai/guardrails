import pytest

from guardrails.cli.hub.deprecation import (
    pip_name_for_uri,
    warn_hub_cli_deprecated,
)


@pytest.mark.parametrize(
    ("uri", "expected"),
    [
        ("hub://guardrails/detect_pii", "guardrails-ai-detect-pii"),
        ("hub://guardrails/regex_match", "guardrails-ai-regex-match"),
        # cross-namespace hub ids still translate to the guardrails-ai-* dist
        ("hub://cartesia/mentions_drugs", "guardrails-ai-mentions-drugs"),
    ],
)
def test_pip_name_for_uri(uri, expected):
    assert pip_name_for_uri(uri) == expected


def test_pip_name_for_uri_invalid_returns_none():
    assert pip_name_for_uri("not-a-hub-uri") is None


def test_warn_hub_cli_deprecated_emits_deprecation_warning():
    with pytest.warns(DeprecationWarning, match="deprecated"):
        warn_hub_cli_deprecated(pip_hint="pip install guardrails-ai-detect-pii")


def test_install_cli_emits_deprecation_and_translates_to_pip(mocker):
    """`guardrails hub install` warns and points at the public pip command."""
    mock_install_multiple = mocker.patch("guardrails.hub.install.install_multiple")
    mocker.patch("guardrails.cli.hub.install.version_warnings_if_applicable")
    mock_warn = mocker.patch("guardrails.cli.hub.deprecation.warn_hub_cli_deprecated")

    from guardrails.cli.hub.install import install_cli

    install_cli(
        ["hub://guardrails/detect_pii"],
        local_models=False,
        quiet=True,
        upgrade=False,
    )

    # The deprecation warning fired with the translated pip command.
    assert mock_warn.call_count == 1
    _, kwargs = mock_warn.call_args
    assert kwargs["pip_hint"] == "pip install guardrails-ai-detect-pii"
    # Install still proceeds (now from public PyPI).
    mock_install_multiple.assert_called_once()
