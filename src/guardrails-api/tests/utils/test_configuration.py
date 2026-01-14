import os
import pytest
from guardrails_api.utils.configuration import valid_configuration, ConfigurationError


def test_valid_configuration(mocker):
    with pytest.raises(ConfigurationError):
        valid_configuration()

    # pg enabled
    os.environ["PGHOST"] = "localhost"
    valid_configuration("config.py")
    os.environ.pop("PGHOST")

    # custom config
    mock_isfile = mocker.patch("os.path.isfile")
    mock_isfile.side_effect = [False, True]
    valid_configuration("config.py")

    # no config
    mock_isfile.side_effect = [False, False]
    with pytest.raises(ConfigurationError):
        valid_configuration("")

    # default config
    mock_isfile = mocker.patch("os.path.isfile")
    mock_isfile.side_effect = [True, False]
    valid_configuration()
