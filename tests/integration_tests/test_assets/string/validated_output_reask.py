from guardrails.utils.reask_utils import FieldReAsk

VALIDATED_OUTPUT_REASK = FieldReAsk(
    incorrect_value="Tomato Cheese Pizza",
    error_message="must be exactly two words",
    fix_value="Tomato Cheese",
)
