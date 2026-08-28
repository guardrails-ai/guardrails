from guardrails import Guard
from guardrails_ai.two_words import TwoWords

# instantiate guards
guard0 = Guard(id="test-guard", name="test-guard").use(TwoWords(on_fail="fix"))
