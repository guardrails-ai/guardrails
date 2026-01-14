"""
All guards defined here will be initialized, if and only if
the application is using in memory guards.

The application will use in memory guards if pg_host is left
undefined. Otherwise, a postgres instance will be started
and guards will be persisted into postgres. In that case,
these guards will not be initialized.
"""

from guardrails import Guard
from guardrails.hub import DetectPII, CompetitorCheck


no_guards = Guard()
no_guards.name = "No Guards"

output_guard = Guard()
output_guard.name = "Output Guard"
output_guard.use_many(
    DetectPII(pii_entities="pii"), CompetitorCheck(competitors=["OpenAI", "Anthropic"])
)
