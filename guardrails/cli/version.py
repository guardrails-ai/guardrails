import contextlib
import requests
import semver
from importlib.metadata import version
from rich.console import Console


GUARDRAILS_PACKAGE_NAME = "guardrails-ai"


def get_guardrails_version():
    return version(GUARDRAILS_PACKAGE_NAME)


def version_warnings_if_applicable(console: Console):
    current_version = get_guardrails_version()

    with contextlib.suppress(Exception):
        res = requests.get(f"https://pypi.org/pypi/{GUARDRAILS_PACKAGE_NAME}/json")
        version_info = res.json()
        info = version_info.get("info", {})
        latest_version = info.get("version")

        is_update_available = semver.compare(latest_version, current_version) > 0

        if is_update_available:
            console.print(
                "[yellow]There is a newer version of Guardrails "
                f"available {latest_version}. Your current version "
                f"is {current_version}[/yellow]!"
            )
