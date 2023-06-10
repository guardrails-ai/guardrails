from guardrails.utils.reask_utils import ReAsk

VALIDATED_OUTPUT_REASK = ReAsk(
    incorrect_value="Tomato Cheese Pizza",
    error_message="must be exactly two words",
    fix_value="Tomato Cheese",
)
