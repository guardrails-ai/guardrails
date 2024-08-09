# Parallelization
## And the Orchestration of Guard Executions

This document is a description of the current implementation of the Guardrails' validation loop.  It attempts to explain the current patterns used with some notes on why those patterns were accepted at the time of implementation and potential future optimizations.  It is _not_ meant to be prescriptive as there can, and will, be improvements made in future versions.

In general you will find that our approach to performance is two fold:
1. Complete computationally cheaper, static checks first and exit early to avoid spending time and resources on more expensive checks that are unlikely to pass when the former fail.
2. Parallelize processing where possible.

## Background: The Validation Loop
When a Guard is executed, that is called via `guard()`, `guard.parse()`, `guard.validate()`, etc., it goes through an internal process that has the following steps:

1. Call the LLM
2. Parse the output
3. Validate the output
4. Reask if necessary

Each of these steps can potentially have sub-steps as well as breakpoints to end the validation loop early if future steps cannot be sucessfully or meaningfully run.

### Calling the LLM
If a Guard is called with an `llm_output`, this step is essentially a no-op as we just return the passed in value in the same format we would have expected it from the LLM.

However, if no `llm_output` is provided but instead an `llm_api` is, then the LLM is called with as many of the arguments passed through as transparently as possible.  If an exception is raised or if the LLM does not return the proper content type, then the validation loop exits early raising a `PromptCallableException`.

    > NOTE: `PromptCallableException` used to serve the purpose of enabling retries.  As of v0.4.5 though, we only attempt retries for OpenAI 0.x.  Since we drop support for 0.x in v0.5.0, we no longer attempt retries, making `PromptCallableException` obsolete.  We should therefore consider its removal.

### Parsing the LLM Output
Regardless if the output was provided by the user or by calling an LLM, it undergoes three distinct parsing steps before validation:

1. Extraction - This step attempts to extract JSON from an LLM response.  It uses a combination of code block detection and regex to retrieve only the JSON content from responses such as:
    ````
    Sure! Here's the JSON you asked for:

    ```
    {
        "foo": "bar"
    }
    ```
    ````
2. Pruning - This step removes extra properties from the response that are not specified by the output structure.  This step _does_ allow for `additionalProperties` specified through JSON Schema, `json_schema_extra`'s in Pydantic, or `<object />` tags without any child elements in RAIL.
3. Type Coercion - This steps attempts to correct any disparities in types between the specified output structure and the LLM output.  For example, if a property is specified as an `int` but the LLM returns `"1"`, this step will convert `"1"` -> `1`.
    > NOTE: This step requires a full traversal of the output schema and LLM output.  While this process is much less expensive than a Reask, we should allow users to disable this step if they wish.

### Validating the LLM Output
Once the output has been parsed it then undergoes validation.  Validation currently happens in two steps:

1. Schema verification - This step ensures that the output matches the schema provided for the output structure.  If the output does not meet the requirements of the schema (e.g. missing properties, unallowed extra properties, incorrect types, etc.), then the validation loop exists early and a `SkeletonReask` is returned.
    > NOTE: This step also requires a full traversal of the output schema and LLM output.  While this process is much less expensive than running validators which utilize large models, we should allow users to disable this step if they wish.
2. Validator execution - This step is likely the most obvious; this is when the validators specified in the Guard are run against the output that has undergone the previous steps.  We will dive into more detail on how these are run below, but in short this step runs the validator and applies the specified on fail actions if the validator results in a `FailResult`.  On fail actions are applied by replacing the value in the output with the appropriate substitute; i.e. a `Filter`, `Refrain`, `FieldReAsk`, or a fixed value.

### Reasking the LLM
The final step in the validation loop is also an optional one.  If you allow Guardrails to make calls to the LLM by providing the `llm_api` argument, and allow reasks by setting `num_reask` to a value greater than zero, then in the event of validation failure a reask prompt is constructed and sent to the LLM with information about what was wrong with it's previous response.  If a reasks are enabled, then the validation loop runs until either validation is sucessful or the maximum number of reasks allowed have been used.

## Async vs Sync
In previous version of Guardrails there was not a strong distinction within the interfaces to specify synchronous or asynchronous flows.  It purely depended on whether the `llm_api` argument was a coroutine or not.  This led to complicated overloads to try to accurately represent the return types as well as ambiguity around what was happening during validation.  In v0.4.5, we introduced the `AsyncGuard` class, and in v0.5.0 we are removing asynchronous support from the `Guard` class entirely in favor of using the `AsyncGuard` class for this purpose.

### Benefits of AsyncGuard
Besides handling asynchronous calls to the LLM, using an `AsyncGuard` also ensures that validation occurs in an asynchronos manner.  A `Guard`, on the other hand, will only utilize asynchronous validation under the following conditions:
* The `GUARDRAILS_PROCESS_COUNT` environment variable is unset, or set to a integer value greater than 1.
* An asyncio event loop is available.
* The asyncio event loop is not taken/already running.

## Validation Orchestration and Parallelization

### Structured Data Validation
We perform validation with a "deep-first" approach.  This has no meaning for unstructured text output since there is only one value, but for structured output it means that the objects are validated from the inside out.

Take the below structure as an example:
```json
{
    "foo": {
        "baz": 1,
        "bez": 2
    },
    "bar": {
        "biz": 1,
        "buz": 2
    }
}
```

As of versions v0.4.x and v0.5.x of Guardrails, the above object would validated as follows:

1. foo.baz
2. foo.bez
3. foo
4. bar.biz
5. bar.buz
6. bar

    > NOTE: The approach currently used, and outlined above, was predicated on the assumption that if child properties fail validation, it is unlikely that the parent property would pass.  With the current atomic state of validation, it can be argued that this assumption is false.  That is, the types of validations applied to parent properties typically take the form of checking the appropriate format of the container like a length check on a list. These types of checks are generally independent of any requirements the child properties have.  This opens up the possibility of running all six paths listed above in parallel at once instead of performing them in steps based on key path.

When synchronous validation occurs as defined in [Benefits of AsyncGuard](#benefits-of-async-guard), the validators for each property would be run in the order they are defined on the schema.  That also means that any on fail actions are applied in that same order.

When asynchronous validation occurs, there are multiple levels of parallelization possible.  First, running validation on the child properties (e.g. `foo.baz` and `foo.bez`) will happen in parallel via the asyncio event loop.  Second, within the validation for each property, if the validators have `run_in_separate_process` set to `True`, they are run in parallel via multiprocessing.  This multiprocessing is capped to the process count specified by the `GUARDRAILS_PROCESS_COUNT` environment variable which defaults to 10.  Note that some environments, like AWS Lambda, may not support multiprocessing in which case you would need to set this environment variable to 1.

### Unstructured Data Validation
When validating unstructured data, i.e. text, the LLM output is treated the same as if it were a property on an object.  This means that the validators applied to is have the ability to run in parallel utilizing multiprocessing when `run_in_separate_process` is set to `True` on the validators.