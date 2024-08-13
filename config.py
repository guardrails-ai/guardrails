from guardrails import Guard
from guardrails.hub import ToxicLanguage
guard = Guard()
guard.name = 'toxic_language_guard'
print("GUARD PARAMETERS UNFILLED! UPDATE THIS FILE!")  # TODO: Remove this when parameters are filled.
guard.use(ToxicLanguage())  # TODO: Add parameters.