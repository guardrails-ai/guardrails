from contextlib import contextmanager
from string import Template
from typing import Callable

from guardrails.hub.validator_package_service import ValidatorPackageService
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
    install_local_models_confirm: Callable = default_local_models_confirm,
):
    """
    Install a validator package from a hub URI.

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
            module_manifest, site_packages, quiet=quiet, logger=cli_logger
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
    validators = ValidatorPackageService.get_validator_from_manifest(module_manifest)

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

    return validators
