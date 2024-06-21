# Response Formatting and Structure

Guardrails provides several interfaces to help llms respond in valid JSON which can then be validated using Guardrails validators.

## Prerequisites

First, create a Pydantic model and guard for the structured data. The model should describe the data structure to be returned. Fields do support  We've included here a sample prompt and data to extract the structured data from. 

```py
from guardrails import Guard
from guardrails.hub import RegexMatch
from pydantic import BaseModel, Field
from typing import List

NAME_REGEX = "^[A-Z][a-z]+\s[A-Z][a-z]+$"

class Delivery(BaseModel):
    custome_name: str=Field(validators=[RegexMatch(regex=NAME_REGEX)], description="customer name")
    pickup_time: str=Field(description="date and time of pickup")
    pickup_location: str=Field(description="address of pickup")
    dropoff_time: str=Field(description="date and time of dropoff")
    dropoff_location: str=Field(description="address of dropoff")
    price: str = Field(description="price of delivery with currency symbol included")

class Schedule(BaseModel):
    deliveries: List[Delivery]

guard = Guard.from_pydantic(Schedule)
chat_history="""
nelson and murdock: i need a pickup 797 9th Avenue, manila envelope, June 3 10:00am with dropoff 10:30am Courthouse, 61 Center Street C/O frank james
operator: quote - $23.00
neslon and murdock: perfect, we accept the quote
operator: 797 9th ave, 10:00am pickup comfirmed
abc flowers: i need a pickup of a flowers from abc flowers at 21 3rd street at 11:00am on june 2 with a dropoff at 75th Ave at 5:30pm same day
operator: 21 3rd street flowers quote - $14.50
abc flowers: accepted
polk and wardell: i need a pickup of a bagels from Bakers Co at 331 5th street at 11:00am on june 3 with a dropoff at 75th Ave at 5:30pm same day
operator: 331 5th street bagels quote - $34.50
polk and wardell: accepted
"""

prompt = """
From the chat exchanges below extract a schedule of deliveries.
Chats:
${chat_history}
"""
```

## Function/tool calling

For models that support openai tool/function calling(`gpt-4o`, `gpt-4-turbo`, or `gpt-3.5-turbo`)

```py
tools = [] # an open ai compatible list of tools

response = guard(
    openai.chat.completions.create,
    model='gpt-4o',
    instructions='You are a helpful assistant.',
    prompt=prompt,
    prompt_params={"chat_history": chat_history},
    tools=tools,
    tool_choice="required",
)
```

## Prompt Updates

For models that support JSON through prompt engineering and hinting(most models)
```py
prompt+="""

${gr.complete_json_suffix_v3}
"""
response = guard(
    openai.chat.completions.create,
    model='gpt-4o',
    instructions='You are a helpful assistant.',
    prompt=prompt,
    prompt_params={"chat_history": chat_history},
)
```

## JSON Mode
For models that support JSON through an input argument(`gpt-4o`, `gpt-4-turbo`, or `gpt-3.5-turbo`)

```py
prompt+="""

${gr.complete_json_suffix_v3}
"""
response = guard(
    openai.chat.completions.create,
    model='gpt-4o',
    instructions='You are a helpful assistant.',
    prompt=prompt,
    prompt_params={"chat_history": chat_history},
    response_format={ "type": "json_object" }
)
```

## Constrained decoding generation


