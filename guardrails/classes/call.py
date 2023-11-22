from typing import Optional
from pydantic import Field
from guardrails.classes.call_inputs import CallInputs
from guardrails.classes.iteration import Iteration
from guardrails.classes.outputs import Outputs
from guardrails.classes.stack import Stack


class Call(Iteration):
    iterations: Stack[Iteration] = Field(
        description="A stack of iterations for each step/reask that occurred during this call.",
        default_factory=Stack
    )
    inputs: CallInputs = Field(
        description="The inputs as passed in to Guard.__call__ or Guard.parse",
        default_factory=CallInputs
    )
    
    # We might just spread these properties instead of containering them
    @property
    def outputs(self) -> Outputs:
        """The outputs from the last iteration."""
        last_iteration = self.iterations.last
        if last_iteration:
            return last_iteration.outputs
        # To allow chaining without getting AttributeErrors
        return Outputs()

    # TODO    
    # @property
    # def logs(self) -> Stack[str]:
    #     """Returns all logs from all iterations as a stack"""

    @property
    def tokens_consumed(self) -> Optional[int]:
        """Returns the total number of tokens consumed during all iterations with this call"""

    @property
    def status(self) -> str:
        """Returns the status of the last iteration"""