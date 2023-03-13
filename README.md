# Guardrails



## XML Files

- define what information you want in the repsonse. Like defining a schema.
- `Schema.from_rail` reads in the `.rail` file and parses it to make sure it's consistent with the dialect (e.g. `.rail` dialect), ensures that the validators are registered for valid formatters. If you pass in formatters that don't exist, then it will raise a warning, but won't create any validation for those formatters. This also creates a prompt that is passed to the LLM

### Formatter
`two-words` is a formatter

### Validator


### Event Handlers
`on:fail:two-words={action}` is an event handler that takes a particular action when the two-words formatter fails.


```xml
<?xml version="1.0"?>
<prompt>
    <list name="fees" description="What fees and charges are associated with my account?">
        <object>
            <integer name="index" format="1-indexed" />
            <string name="name" format="lower-case; two-words" />
            <string name="explanation" format="one-line" />
            <float name="value" format="percentage" />
            <string name="description" format="length: 0 200" />
            <string name="example" required="True" format="tone-twitter explain-high-quality" />
            <string name="advertisement" format="tagline tv-ad" />
        </object>
    </list>
    <string name='interest_rates' description='What are the interest rates offered by the bank on savings and checking accounts, loans, and credit products?'/>
    <!-- <string name='fees' description='What fees and charges are associated with my account?' format="max-len: 5; explain-like-im-five; valid-choices: {[0,5,10]}"/> -->
</prompt>
```