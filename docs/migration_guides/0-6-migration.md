# Migrating to 0.6.0

## New Features

### Messages support for reask and RAILS
`Guard.__call` and rails now fully support `reask_messages` as an argument.

### For `Guard.__call`
```py
response = guard(
    messages=[{
        "role":"user",
        "content":"tell me a joke"
    }],
    reask_messages=[{
        "role":"system"
        "content":"your previous joke failed validation can you tell me another joke?"
    }]
)
```

#### For Rail
```
<rail version="0.1">
<messages>
    <message role="system">
        Given the following document, answer the following questions. If the answer doesn't exist in the document, enter 'None'.

        ${document}

        ${gr.xml_prefix_prompt}
    </message>
    <message role="user">
        ${question}
    </message>
</messages>

<reask_messages>
    <message="system">
        You were asked ${question} and it was not correct can you try again?
    </message>
</reask_messages>
</rail>
```

## Improvements

### Per message validation and fix support
Message validation is now more granular and executed on a per message content basis.
This allows for on_fix behavior to be fully supported.

## Backwards-incompatible changes
### Validator onFail default behavior is now exception
Previously the default behavior for validation failure was noop. This meant developers were required to set it on_fail to exception or check validation_failed
to know if validation failed. This was unintuitive for new users and led to confusion around if validators were working or not. This new behavior will require 
exception handling be added or configurations manually to be set to noop if desired. 

### Simplified schema injection behavior
Previously prompt and instruction suffixes and formatting hints were sometimes automatically injected
into prompt and instructions if guardrails detected xml or a structured schema being used for output. 
This caused confusion and unexpected behavior when arguments to llms were being mutated without a developer asking for it.
Developers will now need to intentionally include Guardrails template variables such as `${gr.complete_xml_suffix_v2}`

### Guardrails Server migration from Flask to FastAPI
In 0.5.X guard.__call and guard.validate async streaming received support for on_fail="fix" merge and parallelized async validation.
We have updated the server to use FastAPI which is build on ASGI to be able to fully take advantage of these improvements.
config.py now fully supports the definition of AsyncGuards and streaming=true as a request argument.
We recommend the combination of `gunicorn` and `uvicorn.workers.UvicornWorker`s

### Streamlined prompt, instructions and msg_history arguments into messages
`Guard.__call` prompt, instruction, reask_prompt and reask_instruction arguments have been streamlined into messages and reask_messages
Instructions should be specified with the role system and prompts with the role user. Any of the out of the box supported llms that require only one text prompt will automatically have the messages converted to one unless a custom callable is being used.
```
# version < 0.6.0
guard(
    instructions="you are a funny assistant",
    prompt="tell me a joke"
)

# version >= 0.6.0
guard(
    messages=[
        {"role":"system", "content":"you are a funny assistant"},
        {"role":"user", "content":"tell me a joke"}
    ]
)
```

### Removal of guardrails OpenAI, Cohere, Anthropic Callables
These callables are being removed in favor of support through passing no callable and setting the appropriate api key and model argument. See LINK TO DOCS for more info.

### Prompt no longer a required positional argument on custom callables
Custom callables will no longer throw an error if the prompt arg is missing in their declaration and guardrails will no longer pass prompt as the first argument. They need to be updated to the messages kwarg to get text input. If a custom callables underlying llm only accepts a single string a helper exists that can compose messages into one otherwise some code to adapt them will be required. 

```py
from guardrails import messages_to_prompt_string

class CustomCallableCallable(PromptCallableBase):
    def llm_api(
        self,
        *args,
        **kwargs,
    ) -> str:
        messages = kwargs.pop("messages", [])
        prompt = messages_to_prompt_string(messages)

        llm_string_output = some_llm_call_requiring_prompt(
            prompt,
            *args,
            **kwargs,
        )
        return llm_string_output
````