import logging
import os

os.environ["COLOREDLOGS_LEVEL_STYLES"] = (
    "spam=white,faint;success=green,bold;debug=magenta;verbose=blue;notice=cyan,bold;warning=yellow;error=red;critical=background=red"  # noqa
)
LEVELS = {
    "SPAM": 5,
    "VERBOSE": 15,
    "NOTICE": 25,
    "SUCCESS": 35,
}
for key in LEVELS:
    logging.addLevelName(LEVELS.get(key), key)  # type: ignore


logger = logging.getLogger("guardrails-cli")
