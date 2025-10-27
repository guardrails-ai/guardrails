# Custom Validators

If you need to perform a validation that is not currently supported by the hub, you can create your own custom validators.

## As A Function
A custom validator can be as simple as a single function if you do not require additional arguments:

```py
from typing import Dict
from guardrails.validators import (
    FailResult,
    PassResult,
    register_validator,
    ValidationResult,
)

@register_validator(name="toxic-words", data_type="string")
def toxic_words(value, metadata: Dict) -> ValidationResult:
    mentioned_words = []
    for word in ["butt", "poop", "booger"]:
        if word in value:
            mentioned_words.append(word)

    if len(mentioned_words) > 0:
        return FailResult(
            error_message=f"Mention toxic words: {', '.join(mentioned_words)}",
        )
    else:
        return PassResult()
```

## As A Class
If you need to perform more complex operations or require additional arguments to perform the validation, then the validator can be specified as a class that inherits from our base Validator class:

```py
from typing import Callable, Dict, Optional, List
from guardrails.validators import (
    FailResult,
    PassResult,
    register_validator,
    ValidationResult,
    Validator,
)

@register_validator(name="toxic-words", data_type="string")
class ToxicWords(Validator):
    def __init__(self, search_words: List[str]==["booger", "butt"], on_fail: Optional[Callable] = None):
        super().__init__(on_fail=on_fail, search_words=search_words)
        self.search_words = search_words

    def _validate(self, value: List[str], metadata: Dict) -> ValidationResult:
        mentioned_words = []
        for word in self.search_words:
            if word in value:
                mentioned_words.append(word)

        if len(mentioned_words) > 0:
            return FailResult(
                error_message=f"Mentioned toxic words: {', '.join(mentioned_words)}",
            )
        else:
            return PassResult()
```

## On Fail

In the code below, a `fix_value` will be supplied in the `FailResult`. This value will represent a programmatic fix that can be applied to the output if `on_fail='fix'` is passed during validator initialization.

> For more details about on fail actions refer to: [On Fail Actions](/concepts/validator_on_fail_actions)

```py
from typing import Callable, Dict, Optional
from guardrails.validators import (
    FailResult,
    PassResult,
    register_validator,
    ValidationResult,
    Validator,
)

@register_validator(name="toxic-words", data_type="string")
class ToxicWords(Validator):
    def __init__(self, search_words: str[]=["booger", "butt"], on_fail: Optional[Callable] = None):
        super().__init__(on_fail=on_fail, search_words=search_words)
        self.search_words = search_words

    def _validate(self, value: str, metadata: Dict) -> ValidationResult:
        mentioned_words = []
        for word in self.search_words:
            if word in value:
                mentioned_words.append(word)

        if len(mentioned_words) > 0:
            # Filter out the toxic words from the value
            on_fix = ' '.join([word for word in value.split() if word not in self.search_words])
            return FailResult(
                error_message=f"Value {value} does mention words: {', '.join(mentioned_words)}",
                on_fix=on_fix,
            )
        else:
            return PassResult()
```

## Model and LLM Integration

Validators are not limited to just programmatic and algorithmic validation. 

In this example we integrate with Hugging Face to validate if a value meets a threshold for toxic language.
```py
from typing import Callable, Dict, Optional
from guardrails.validators import (
    FailResult,
    PassResult,
    register_validator,
    ValidationResult,
    Validator,
)
from litellm import completion
from transformers import pipeline

@register_validator(name="toxic-language", data_type="string")
class ToxicLanguageValidator(Validator):
    def __init__(
            self, 
            threshold: float = 0.9,
            device: int = -1,  # Add device parameter with default value -1 (CPU). 0 is GPU.
            model_name: str = "unitary/toxic-bert",
            on_fail: Optional[Callable] = None
            ):
        super().__init__(on_fail=on_fail, threshold=threshold)
        self._threshold = threshold
        self.pipeline = pipeline("text-classification", model=model_name, device=device)

    def _validate(self, value: str, metadata: Dict) -> ValidationResult:
        result = self.pipeline(value)
        if result[0]['label'] == 'toxic' and result[0]['score'] > self._threshold:
            return FailResult(
                error_message=f"{value} failed validation. Detected toxic language with score {result[0]['score']}."
            )
        else:
            return PassResult()
```

The validator below can similarly identify toxic language but instead of using model trained for it specifically it uses chat-gpt.
```py
from typing import Callable, Dict, Optional
from guardrails.validators import (
    FailResult,
    PassResult,
    register_validator,
    ValidationResult,
    Validator,
)
from litellm import completion

PROMPT = """
You are a helpful assistant.
Analyze the other messages provided and respond with a number between 0 and 100 and nothing else.
Do not output prose.
0 means the messages contain no toxic language.
100 means the messages contain very toxic language that is offensive, vulgar, negative or insulting.
"""

@register_validator(name="toxic-language", data_type="string")
class ToxicLanguage(Validator):
    def __init__(self, threshold: int=70, model='gpt-4o', on_fail: Optional[Callable] = None):
        super().__init__(on_fail=on_fail, threshold=threshold)
        self._threshold = threshold
        self._model = model

    def _llm_callable(self, messages):
        return completion(
            model=self._model,
            messages=messages,
            temperature=0.0,
        )
    
    def _validate(self, value: str, metadata: Dict) -> ValidationResult:
        messages = [
            {
                "role": "system",
                "content": PROMPT,
            },
            {
                "role": "user",
                "content": value,
            }
        ]

        score = int(self._llm_callable(messages).choices[0].message.content)
        if score > self._threshold:
            return FailResult(
                error_message=f"{value} failed validation. Score {score} is below threshold of {self._threshold}."
            )
        else:
            return PassResult()
```

## Streaming

Validators support streaming validation out of the box. The validate_stream method handles calling _validate with accumulated chunks of a stream when a guard is executed with `guard(streaming=True, ...)`

By default stream validation is done on a per sentence basis. Validator._chunking_function may be overloaded to provide a custom chunking strategy. This is useful to optimize latency when integrating outside services such as llms and controlling how much data an validating model gets to give it more or less context.

The code below in a validator will cause a validator to validate a stream of text 1 paragraph at a time.

```py
@register_validator(name="toxic-language", data_type="string")
class ToxicLanguageValidator(Validator):
    def __init__(
            self, 
            threshold: float = 0.9,
            device: int = -1,  # Add device parameter with default value -1 (CPU). 0 is GPU.
            model_name: str = "unitary/toxic-bert",
            on_fail: Optional[Callable] = None
            ):
        super().__init__(on_fail=on_fail, threshold=threshold)
        self._threshold = threshold
        self.pipeline = pipeline("text-classification", model=model_name, device=device)

    def _validate(self, value: str, metadata: Dict) -> ValidationResult:
        result = self.pipeline(value)
        if result[0]['label'] == 'toxic' and result[0]['score'] > self._threshold:
            return FailResult(
                error_message=f"{value} failed validation. Detected toxic language with score {result[0]['score']}."
            )
        else:
            return PassResult()

    def _chunking_function(chunk: str) -> List[str]:
        """The strategy used for chunking accumulated text input into
        validation sets of a paragraph

        Args:
            chunk (str): The text to chunk into some subset.

        Returns:
            list[str]: The text chunked into some subset.
        """
        if "\n\n" not in chunk:
            return []
        fragments = chunk.split("\n\n")
        return [fragments[0] + "\n\n", "\n\n".join(fragments[1:])]
        
        return chunks
```

## Usage

Custom validators must be defined before creating a `Guard` or `RAIL` spec in the code, 
but otherwise can be used like built in validators. It can be used in a `RAIL` spec OR
a `Pydantic` model like so:

Custom validators must be defined before creating a `Guard` in the code, 
but otherwise can be used just like built in validators.

### Guard.use Example
```py
from guardrails import Guard
from .my_custom_validators import toxic_words, ToxicLanguage

guard = Guard().use(
    ToxicLanguage(threshold=0.8)
).use(
    toxic_words()
)
```

### Pydantic Example
```py
from guardrails import Guard
from pydantic import BaseModel, Field
from .my_custom_validators import toxic_words, ToxicLanguage

class MyModel(BaseModel):
    a_string: Field(validators=[toxic_words()])
    custom_string: Field(validators=[ToxicLanguage(threshold=0.8)])

guard = Guard.for_pydantic(MyModel)
```

### RAIL Example
```xml
<rail version="0.1">
...
<output>
    <string name="a_string" type="string" validators="toxic-words" on-fail-toxic-words="exception" />
    <string name="custom_string" type="string" validators="toxic-language:threshold=0.4" />
</output>
...
</rail>
``` 