# Guardrails.ai

Guardrails is a python package supporting:

1. `RAIL` specification for defining the expected outcome to the LLM.
2. Lightweight wrappers around LLM API calls that allow output validation and correction.

Let's say you want to extract key information from a Terms-of-Service document. Here's how you can use Guardrails to get a structured, validated and corrected output from the LLM.

## Step 1: Add expected output schema to the `RAIL` spec.
Start by specifying the schema and format of their desired output in an .rail file.

```xml
<rail version=0.1>
<output>
    <list name="fees" description="What fees and charges are associated with my account?">
        <object>
            <integer name="index" format="1-indexed" />
            <string name="name" format="lower-case; two-words" on-fail-lower-case="noop" on-fail-two-words="reask"/>
            <string name="explanation" format="one-line" on-fail-one-line="noop" />
            <float name="value" format="percentage"/>
            <string name="description" format="length: 0 200" on-fail-length="noop" />
            <string name="example" required="True" format="tone-twitter explain-high-quality" />
            <string name="advertisement" format="tagline tv-ad" />
        </object>
    </list>
    <string name='interest_rates' description='What are the interest rates offered by the bank on savings and checking accounts, loans, and credit products?' format="one-line" on-fail-one-line="noop"/>
</output>
</rail>
```

## Step 2: Add the information about the high level task to the `RAIL` spec.

Add the prompt to the .rail file. .rail supports templating, and provides helpful primitives out-of-the-box to aid in prompt construction.

<rail version="0.1">
<prompt>

Given the following document, answer the following questions. If the answer doesn't exist in the document, enter 'None'.

{document}

@xml_prefix_prompt

{{output_schema}}

@json_suffix_prompt
</prompt>

<output>
	...
</output>

</rail>

At runtime, the `{{output_schema}}` specification will be substituted automatically by the correct value. Anything enclosed in `{}` is a prompt variable which will be substituted at runtime. `@xml_prefix_prompt` and `@json_suffix_prompt` are guardrails primitives.

## Step 3: Wrap the LLM API call with Guardrails.

Wrap any LLM API call with Guardrails to make sure that the generated output is checked, validated and corrected.

```python
import guardrails as gd
import openai

guard = gd.Guard.from_rail("path/to/rail/file")
output = guard(
		openai.Completion.create,
		document=document,
		temperature=0.0,
		num_tokens=1024
)
```

That’s it! Running this code snippet returns the following output:

<!-- Add click to expand markdown block below: -->
<details>
```json
{
		'interest_rates': 'Purchase Annual Percentage Rate (APR) 0% Intro APR for the first 18 months that your Account
is open. After that, 19.49%. This APR will vary with the market based on the Prime Rate. My Chase Loan SM APR 
19.49%. This APR will vary with the market based on the Prime Rate. Balance Transfer APR 0% Intro APR for the first
18 months that your Account is open. After that, 19.49%. This APR will vary with the market based on the Prime 
Rate. Cash Advance APR 29.49%. This APR will vary with the market based on the Prime Rate. Penalty APR and When It 
Applies Up to 29.99%. This APR will vary with the market based on the Prime Rate.'
    'fees': [
        {
            'index': 1,
            'name': 'annusal membership',
            'explanation': 'annual membership fee',
            'description': 'None',
            'advertisement': 'None'
        },
        {
            'index': 2,
            'name': 'my chase',
            'explanation': 'fixed finance charge',
            'description': 'Monthly fee of 0% of the amount of each eligible purchase transaction or amount 
selected to create a My Chase Plan while in the 0% Intro Purchase APR period. After that, monthly fee of 1.72% of 
the amount of each eligible purchase transaction or amount selected to create a My Chase Plan. The My Chase Plan 
Fee will be determined at the time each My Chase Plan is created and will remain the same until the My Chase Plan 
is paid in full.',
            'advertisement': 'None'
        },
        {
            'index': 3,
            'name': 'balance transfers',
            'explanation': 'intro fee',
            'description': 'Intro fee of either $5 or 3% of the amount of each transfer, whichever is greater, on 
transfers made within 60 days of account opening. After that: Either $5 or 5% of the amount of each transfer, 
whichever is greater.',
            'advertisement': 'None'
        },
        {
            'index': 4,
            'name': 'cash advances',
            'explanation': 'transaction fee',
            'description': 'Either $10 or 5% of the amount of each transaction, whichever is greater.',
            'advertisement': 'None'
        },
        {
            'index': 5,
            'name': 'foreign transactions',
            'explanation': 'transaction fee',
            'description': '3% of the amount of each transaction in U.S. dollars.',
            'advertisement': 'None'
        },
        {
            'index': 6,
            'name': 'late payment',
            'explanation': 'penalty fee',
            'description': 'Up to $40.',
            'advertisement': 'None'
        },
        {
            'index': 7,
            'name': 'over-the',
            'explanation': 'penalty fee',
            'description': 'None',
            'advertisement': 'None'
        },
        {
            'index': 8,
            'name': 'return payment',
            'explanation': 'penalty fee',
            'description': 'Up to $40.',
            'advertisement': 'None'
        },
        {
            'index': 9,
            'name': 'return check',
            'explanation': 'penalty fee',
            'description': 'None',
            'advertisement': 'None'
        }
    ]
}
```
</details>