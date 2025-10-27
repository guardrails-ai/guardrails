# Frequently Asked Questions

## I get an "Unauthorized" error when installing validators from the Guardrails Hub. What should I do?

If you see an "Unauthorized" error when installing validators from the Guardrails hub, it means that the API key you are using is not authorized to access the Guardrails hub. It may be unset or expired. 

To fix this, first generate a new API key from the [Guardrails Hub](https://hub.guardrailsai.com/keys). Then, configure the Guardrails CLI with the new API key.

```bash
guardrails configure
```

There is also a headless option to configure the CLI with the token.

```bash
guardrails configure --token <your_token>
```

## I'm seeing a PromptCallableException when invoking my Guard. What should I do?

If you see an exception that looks like this

```
PromptCallableException: The callable `fn` passed to `Guard(fn, ...)` failed with the following error: `custom_llm_func() got an unexpected keyword argument 'messages'`. Make sure that `fn` can be called as a function that takes in a single prompt string and returns a string.
```

It means that the call to the LLM failed. This is usually triggered for one of the following reasons:

1. An API key is not present or not passed correctly to the LLM
1. The LLM API was passed arguments it doesn't expect. Our recommendation is to use the LiteLLM standard, and pass arguments that conform to that standard directly in the guard callable. It's helpful as a debugging step to remove all other arguments or to try and use the same arguments in a LiteLLM client directly.
1. The LLM API is down or experiencing issues. This is usually temporary, and you can use LiteLLM or the LLM client directly to verify if the API is working as expected.
1. You passed a custom LLM callable, and it either doesn't conform to the expected signature or it throws an error during execution. Make sure that the custom LLM callable can be called as a function that takes in messages kwarg and returns a string.

## How can I host Guardrails as its own server

Guardrails can run totally on the server as of version 0.5.0. You can use the `guardrails-ai` package to run Guardrails as a server. You can find more information on how to do this in the [Guardrails AI documentation](https://docs.guardrails.ai/guardrails_ai/installation).

## Which validators should I use? Where can I find them?

You can find a variety of validators on the [Guardrails Hub](https://hub.guardrailsai.com). We recommend starting by drilling down into your usecase on the left nav of that page (chatbot, customer support, etc...). We're also coming out with starter packs soon that are generally applicable to common usecases.

## How does Guardrails impact my LLM app's latency

tl;dr - guardrails aims to add < 100ms to each LLM request, use our recommendations to speed stuff up.

We've done a lot of work to make Guardrails perform well. Validating LLMs is not trivial, and because of the different approaches used to solve each kind of validation, performance can vary. Performance can be split into two categories: Guard execution and Validation execution. Guard execution is the time it takes to amend prompts, parse LLM output, delegate validation, and compile validation results. 

Guard execution time is minimal, and should run on the order of microseconds.

Validation execution time is usually on the order of tens of milliseconds. There are definitely standouts here. For example, some ML-based validators like [RestrictToTopic](https://hub.guardrailsai.com/validator/tryolabs/restricttotopic) can take seconds to run when the model is cold and running locally on a CPU. However, we've seen it run in sub-100ms time when the model is running on GPUs.

Here are our recommendations:

1. use streaming when presenting user-facing applications. Streaming allows us to validate smaller chunks (sentences, phrases, etc, depending on the validator), and this can be done in parallel as the LLM generates the rest of the output
1. Host your validator models on GPUs. Guardrails provides inference endpoints for some popular validators, and we're working on making this more accessible.
1. Run Guardrails on its own dedicated server. This allows the library to take advantage of a full set of compute resources to thread out over. It also allows you to scale Guardrails independently of your application
1. In production and performance-testing environments, use telemetry to keep track of validator latency and how it changes over time. This will help you understand right-sizing your infrastructure and identifying bottlenecks in guard execution.

## How do I setup my own `fix` function for validators in a guard?

If we have a validator that looks like this
```python
from guardrails.validators import PassResult, FailResult, register_validator

@register_validator(name="is_cake", data_type="string")
def is_cake(value, metadata):
    if value == "cake":
        return PassResult()
    return FailResult(error_message="This is not a cake.")
```

You can override the `fix` behavior by passing it as a function to the Guard object when the validator is declared.

```python
from guardrails import Guard

def fix_is_cake(value, fail_result: FailResult):
    return "IT IS cake"

guard = Guard().use(is_cake, on_fail=fix_is_cake)

res = guard.parse(
    llm_output="not cake"
)

print(res.validated_output) # Validated outputs
# Prints "IT IS cake"
```

## I'm encountering an XMLSyntaxError when creating a `Guard` object from a `RAIL` specification. What should I do?

Make sure that you are escaping the `&` character in your `RAIL` specification. The `&` character has a special meaning in XML, and so you need to escape it with `&amp;`. For example, if you have a prompt like this:

```xml
<messages>
<message role="user">
    This is a prompt with an & character.
</message>
</messages>
```

You need to escape the `&` character like this:

```xml
<messages>
<message role="user">
    This is a prompt with an &amp; character.
</message>
</messages>
```

## Are validators all model-based? Are they proprietary to Guardrails?

Some validators are model based, some validators use LLMs, and some validators are purely logic. They are usually not proprietary, you can see a variety of them on the [Guardrails Hub](https://hub.guardrailsai.com) where they are largely open source and licensed under [Apache 2.0](https://www.apache.org/licenses/LICENSE-2.0).

This doesn't preclude the creation and use of custom private validators with any given technology, but we are huge believers in the power of open source software.

## Issues loging in with `guardrails configure`

If you encounter issues logging in using an API key with `guardrails configure`, they may be caused by one of the following:

1. Your API key has been revoked, expired, or is invalid. You can check your existing tokens and create a new one at [https://hub.guardrailsai.com/tokens](https://hub.guardrailsai.com/tokens) if necessary.
2. If you're using Windows Powershell, ensure that you paste the token by right-clicking instead of using the keyboard shortcut `CTRL+V`, as the shortcut may cause caret characters to be pasted instead of the clipboard contents.

If your login issues persist, please check the contents of the ~/.guardrailsrc file to ensure the correct token has been persisted.

## Where do I get more help if I need it?

If you're still encountering issues, please [open an issue](https://github.com/guardrails-ai/guardrails/issues/new) and we'll help you out!

We're also available on [Discord](https://discord.gg/U9RKkZSBgx) if you want to chat with us directly.

## I'm getting an error related to distutils when installing validators.
This can happen on cuda enabled devices in python versions 3.11 and below when a validator indirectly depends on a package that imports distutils.

If you see an error similar to the one below:
```sh
Installing hub://guardrails/nsfw_text...
[   =] Running post-install setup
Device set to use cpu
/home/ubuntu/support/.venv/lib/python3.11/site-packages/_distutils_hack/__init__.py:18: UserWarning: Distutils was imported before Setuptools, but importing Setuptools also replaces the `distutils` module in `sys.modules`. This may lead to undesirable behaviors or errors. To avoid these issues, avoid using distutils directly, ensure that setuptools is installed in the traditional way (e.g. not an editable install), and/or make sure that setuptools is always imported before distutils.
  warnings.warn(
/home/ubuntu/support/.venv/lib/python3.11/site-packages/_distutils_hack/__init__.py:33: UserWarning: Setuptools is replacing distutils.
  warnings.warn("Setuptools is replacing distutils.")
ERROR:guardrails-cli:Failed to import transformers.pipelines because of the following error (look up to see its traceback):
/home/ubuntu/.pyenv/versions/3.11.11/lib/python3.11/distutils/core.py
```

set the following as an environment variable to tell python to use the builtin version of distutils that exists in 3.11 and below:

```sh
export SETUPTOOLS_USE_DISTUTILS="stdlib"
```