from typing import Optional
from pydantic import Field, BaseModel


class GuardExecutionOptions(BaseModel):
    prompt: Optional[str] = Field(defualt=None)
    instructions: Optional[str] = Field(defualt=None)
    reask_prompt: Optional[str] = Field(defualt=None)
    reask_instructions: Optional[str] = Field(defualt=None)
    num_reasks: Optional[int] = Field(defualt=None)
    prompt_schema: Optional[str] = Field(defualt=None)
    instructions_schema: Optional[str] = Field(defualt=None)
    msg_history_schema: Optional[str] = Field(defualt=None)
