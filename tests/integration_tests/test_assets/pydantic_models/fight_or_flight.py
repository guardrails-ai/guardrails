from tests.integration_tests.test_assets.validators.valid_choices import ValidChoices
from pydantic import BaseModel, Field
from typing import Literal, Optional, Union

prompt = """
You are a human in an enchanted forest.
You come across opponents of different types, 
and you should fight smaller opponents and run away from bigger ones.

You run into a ${opp_type}. What do you do?

${gr.complete_json_suffix_v2}"""


class Fight(BaseModel):
    chosen_action: Literal["fight"]
    weapon: str = Field(validators=[ValidChoices(["crossbow", "machine gun"], on_fail="reask")])


class Flight(BaseModel):
    chosen_action: Literal["flight"]
    flight_direction: Optional[str] = Field(
        validators=[ValidChoices(["north", "south", "east", "west"], on_fail="exception")]
    )
    distance: int = Field(validators=[ValidChoices([1, 2, 3, 4], on_fail="exception")])


class FightOrFlight(BaseModel):
    action: Union[Fight, Flight] = Field(discriminator="chosen_action")
