import pytest


@pytest.fixture
def runner():
    from typer.testing import CliRunner

    return CliRunner()
