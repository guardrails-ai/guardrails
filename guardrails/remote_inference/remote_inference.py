from typing import Optional
from guardrails.classes.rc import RC


# TODO: Consolidate with telemetry switches
def get_use_remote_inference(rc: RC) -> Optional[bool]:
    """Load the use_remote_inferencing setting from the rc file.

    Args:
        rc (RC): The rc settings.

    Returns:
        Optional[bool]: The use_remote_inferencing setting, or None if not found.
    """
    try:
        use_remote_inferencing = rc.use_remote_inferencing
        if isinstance(use_remote_inferencing, str):
            return use_remote_inferencing.lower() == "true"
        elif isinstance(use_remote_inferencing, bool):
            return use_remote_inferencing
        else:
            return None
    except AttributeError:
        # If the attribute doesn't exist, return None
        return None
