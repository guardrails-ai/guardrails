from typing import List

from guardrails_api_client import SimpleTypes
from guardrails.types.rail import RailTypes


types_registry: List[str] = [
    *[st.value for st in SimpleTypes],
    *[rt.value for rt in RailTypes],
]
