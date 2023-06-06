from guardrails.utils.reask_utils import ReAsk

VALIDATED_OUTPUT_REASK = ReAsk(
    incorrect_value="sour cream tomata",
    error_message="must be exactly two words",
    fix_value="sour cream",
)
