from dataclasses import dataclass, field
from typing import Any, Dict, List
from guardrails_api_client import ValidatorReference
from guardrails.classes.execution.guard_execution_options import GuardExecutionOptions
from guardrails.classes.output_type import OutputTypes
from guardrails.types.validator import ValidatorMap


@dataclass
class ProcessedSchema:
    """This class is just a container for the various pieces of information we
    extract from the various schema wrappers a user can pass in; i.e. RAIL or
    Pydantic."""

    output_type: OutputTypes = field(default=OutputTypes.STRING)
    validators: List[ValidatorReference] = field(default_factory=list)
    validator_map: ValidatorMap = field(default_factory=dict)
    json_schema: Dict[str, Any] = field(default_factory=dict)
    exec_opts: GuardExecutionOptions = field(default_factory=GuardExecutionOptions)
