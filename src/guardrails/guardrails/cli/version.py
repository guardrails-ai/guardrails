from importlib.metadata import distribution, version
from pathlib import Path

import requests
import semver
from rich.console import Console


GUARDRAILS_PACKAGE_NAME = "guardrails-ai"


def get_guardrails_version():
    return version(GUARDRAILS_PACKAGE_NAME)


def _is_pypi_install() -> bool:
    """Check if package was installed from PyPI (PEP 610)."""
    dist = distribution(GUARDRAILS_PACKAGE_NAME)
    direct_url_path = Path(dist._path) / "direct_url.json"
    # direct_url.json exists for non-PyPI installs (git, local, URL)
    return not direct_url_path.exists()


def version_warnings_if_applicable(console: Console):
    current_version = get_guardrails_version()

    # Skip version check for non-PyPI installs (git, local, editable)
    if not _is_pypi_install():
        return

    try:
        res = requests.get(
            f"https://pypi.org/pypi/{GUARDRAILS_PACKAGE_NAME}/json",
            timeout=5,
        )
        res.raise_for_status()
        version_info = res.json()
        latest_version = version_info.get("info", {}).get("version")

        if latest_version and semver.compare(latest_version, current_version) > 0:
            console.print(
                "[yellow]There is a newer version of Guardrails "
                f"available {latest_version}. Your current version "
                f"is {current_version}[/yellow]!"
            )
    except requests.RequestException:
        # Network issues - skip version check silently
        pass
    except (ValueError, KeyError):
        # Invalid JSON or missing keys - skip version check silently
        pass
