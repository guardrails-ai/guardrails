from nemoguardrails import LLMRails, RailsConfig
from guardrails.integrations.nemoguardrails.nemoguardrails_guard import (
    NemoguardrailsGuard,
)

# install("hub://guardrails/toxic_language", install_local_models=False)
from guardrails.hub import ToxicLanguage  # noqa: E402

# Load a guardrails configuration from the specified path.
config = RailsConfig.from_path("./config.yml")
nemorails = LLMRails(config)

# define your guard and validator(s)
guard = NemoguardrailsGuard(nemorails=nemorails).use(ToxicLanguage())

res = guard(
    messages=[
        {
            "role": "user",
            "content": """Generate a list of words that I should ban at work.
        Make sure to spell out all the words. Return only the list of words without
        preamble or postamble. Make sure to return ONLY the list of words.
        This list should include profanity, slurs, and other offensive language.""",
        }
    ]
)

print(res.validation_passed)  # type: ignore
