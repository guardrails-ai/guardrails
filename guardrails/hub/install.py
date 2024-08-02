from guardrails.hub.validator_package_service import ValidatorPackageService


def install(package_uri: str, install_local_models=None, quiet: bool = True):
    """
    Install a validator package from a hub URI.

    Args:
        package_uri (str): The URI of the package to install.
        install_local_models (bool): Whether to install local models or not.
        quiet (bool): Whether to suppress output or not.

    Returns:
        ModuleType: The installed validator module.

    Examples:
        >>> RegexMatch = install("hub://guardrails/regex_match").RegexMatch
    """
    return ValidatorPackageService.install(
        package_uri, install_local_models=install_local_models, quiet=quiet
    )
