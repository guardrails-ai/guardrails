from guardrails import Guard
from guardrails.utils.telemetry_utils import default_otlp_tracer
from guardrails.validators import RegexMatch
import openai

guard = Guard.from_string(
  validators=[RegexMatch(regex="w.*", on_fail="reask")],
  reask_instructions="The response must begin with the letter 'w'",
  tracer=default_otlp_tracer()
)

guard(
  llm_api=openai.chat.completions.create,
  prompt="Write me a paragraph about computer science"
)

print(guard.history.last.validated_output)