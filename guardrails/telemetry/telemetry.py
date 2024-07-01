from typing import Optional


from guardrails.classes.credentials import Credentials


# TODO: Consolidate with telemetry switches
def get_disable_telemetry(creds: Credentials) -> Optional[bool]:
    """Load the use_remote_inferencing setting from the credentials.

    Args:
        creds (Credentials): The credentials object.

    Returns:
        Optional[bool]: The use_remote_inferencing setting, or None if not found.
    """
    try:
        return not creds.enable_metrics
    except AttributeError:
        # If the attribute doesn't exist, return None
        return None
