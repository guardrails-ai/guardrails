from contextlib import contextmanager
from string import Template
from typing import Callable, cast, List

from guardrails.hub.validator_package_service import (
    ValidatorPackageService,
    ValidatorModuleType,
)
from guardrails.classes.credentials import Credentials

from guardrails.cli.hub.console import console
from guardrails.cli.logger import LEVELS, logger as cli_logger


class LocalModelFlagNotSet(Exception):
    pass


@contextmanager
def do_nothing_context(*args, **kwargs):
    try:
        yield
    finally:
        pass


def default_local_models_confirm():
    raise LocalModelFlagNotSet(
        "The 'install_local_models' keyword argument"
        " must be explicitly set to True or False to continue."
    )


def install(
    package_uri: str,
    install_local_models=None,
    quiet: bool = True,
    upgrade: bool = False,
    install_local_models_confirm: Callable = default_local_models_confirm,
) -> ValidatorModuleType:
    """Install a validator package from a hub URI.

    Args:
        package_uri (str): The URI of the package to install.
        install_local_models (bool): Whether to install local models or not.
        quiet (bool): Whether to suppress output or not.
        install_local_models_confirm (Callable): A function to confirm the
            installation of local models.

    Returns:
        ModuleType: The installed validator module.

    Examples:
        >>> RegexMatch = install("hub://guardrails/regex_match").RegexMatch

        >>> install("hub://guardrails/regex_match);
        >>> import guardrails.hub.regex_match as regex_match
    """

    verbose_printer = console.print
    quiet_printer = console.print if not quiet else lambda x: None

    # 1. Validation
    has_rc_file = Credentials.has_rc_file()
    module_name = ValidatorPackageService.get_module_name(package_uri)

    installing_msg = f"Installing {package_uri}..."
    cli_logger.log(
        level=LEVELS.get("SPAM"),  # type: ignore
        msg=installing_msg,
    )
    verbose_printer(installing_msg)

    # Define Loader for UX purposes
    loader = console.status if not quiet else do_nothing_context

    # 2. Prep Installation
    fetch_manifest_msg = "Fetching manifest"
    with loader(fetch_manifest_msg, spinner="bouncingBar"):
        (module_manifest, site_packages) = (
            ValidatorPackageService.get_manifest_and_site_packages(module_name)
        )

    # 3. Install - Pip Installation of git module
    dl_deps_msg = "Downloading dependencies"
    with loader(dl_deps_msg, spinner="bouncingBar"):
        ValidatorPackageService.install_hub_module(
            module_manifest,
            site_packages,
            quiet=quiet,
            upgrade=upgrade,
            logger=cli_logger,
        )

    use_remote_endpoint = False
    module_has_endpoint = (
        module_manifest.tags and module_manifest.tags.has_guardrails_endpoint
    )

    try:
        if has_rc_file:
            # if we do want to remote then we don't want to install local models
            use_remote_endpoint = (
                Credentials.from_rc_file(cli_logger).use_remote_inferencing
                and module_has_endpoint
            )
        elif install_local_models is None and module_has_endpoint:
            install_local_models = install_local_models_confirm()
    except AttributeError:
        pass

    # 4. Post Installation
    install_local_models = (
        install_local_models if install_local_models is not None else True
    )
    if not use_remote_endpoint and install_local_models is True:
        cli_logger.log(
            level=LEVELS.get("SPAM"),  # type: ignore
            msg="Installing models locally!",
        )
        post_msg = "Running post-install setup"
        with loader(post_msg, spinner="bouncingBar"):
            ValidatorPackageService.run_post_install(
                module_manifest, site_packages, logger=cli_logger
            )
    else:
        cli_logger.log(
            level=LEVELS.get("SPAM"),  # type: ignore
            msg="Skipping post install, models will not be "
            "downloaded for local inference.",
        )
    ValidatorPackageService.add_to_hub_inits(module_manifest, site_packages)

    # 5. Get Validator Class for the installed module
    installed_module = ValidatorPackageService.get_validator_from_manifest(
        module_manifest
    )
    installed_module = cast(ValidatorModuleType, installed_module)

    # Print success messages
    cli_logger.info("Installation complete")

    verbose_printer(f"✅Successfully installed {module_name}!\n\n")
    success_message_cli = Template(
        "[bold]Import validator:[/bold]\n"
        "from guardrails.hub import ${export}\n\n"
        "[bold]Get more info:[/bold]\n"
        "https://hub.guardrailsai.com/validator/${id}\n"
    ).safe_substitute(
        module_name=package_uri,
        id=module_manifest.id,
        export=module_manifest.exports[0],
    )
    success_message_logger = Template(
        "✅Successfully installed ${module_name}!\n\n"
        "Import validator:\n"
        "from guardrails.hub import ${export}\n\n"
        "Get more info:\n"
        "https://hub.guardrailsai.com/validator/${id}\n"
    ).safe_substitute(
        module_name=package_uri,
        id=module_manifest.id,
        export=module_manifest.exports[0],
    )
    quiet_printer(success_message_cli)  # type: ignore
    cli_logger.log(level=LEVELS.get("SPAM"), msg=success_message_logger)  # type: ignore

    # Not a fan of this but allows the installation to be used in create command as is
    installed_module.__validator_exports__ = module_manifest.exports

    return installed_module


def install_multiple(
    package_uris: List[str],
    install_local_models=None,
    quiet: bool = True,
    upgrade: bool = False,  # Add the upgrade parameter here
    install_local_models_confirm: Callable = default_local_models_confirm,
) -> List[ValidatorModuleType]:
    """Install multiple validator packages from hub URIs.

    Args:
        package_uris (List[str]): List of URIs of the packages to install.
        install_local_models (bool): Whether to install local models or not.
        quiet (bool): Whether to suppress output or not.
        upgrade (bool): Whether to upgrade to the latest package version.
        install_local_models_confirm (Callable): A function to confirm the
            installation of local models.

    Returns:
        List[ValidatorModuleType]: List of installed validator modules.
    """
    installed_modules = []

    for package_uri in package_uris:
        installed_module = install(
            package_uri,
            install_local_models=install_local_models,
            quiet=quiet,
            upgrade=upgrade,  # Pass upgrade here
            install_local_models_confirm=install_local_models_confirm,
        )
        installed_modules.append(installed_module)

    return installed_modules
