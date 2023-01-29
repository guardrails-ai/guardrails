# Getting Started

In this notebook, we will go through the basics of creating and `AIML` spec and using Guardrails to enforce it.

## Objective

Our goal is understand what a bank run is, and generate URL links to relevant news articles. We will first generate an `AIML` spec for this and then use Guardrails to enforce it.

## Installation

To get started, install the `guardrails` package with `pip`.



```python
!pip install guardrails-ai
```

## Creating an `AIML` spec

At the heart of `Guardrails` is the `AIML` spec.

`AIML` a flavor of XML (standing for `AI Markup Language`) that describes the expected structure and type of the output of the LLM, the quality criteria for the output to be valid and corrective actions to be taken if the output is invalid.

- For this task, we create an `AIML` spec that requests the LLM to generate an object with two fields: `explanation` and `follow_up_url`.
- For the `explanation` field to be valid, the max length of the generated string should be 280 characters. In case the generated string is between **200 to 280 characters in length**.
- For the `follow_up_url` field to be valid, the URL should be reachable. In case the **URL is not reachable**, the LLM should be reasked to generate a valid URL.

Ordinarily, the `AIML` spec would be created in a file directly. However, for the purposes of this demo, we write the spec in a string and then create a file from it.

We specify our quality criteria (generated length, URL reachability) in the `format` fields of the `AIML` spec below. For now, we want to do nothing if the quality criteria for `explanation` is not met, and filter the `follow_up_url` if it is not valid.


```python
aiml_spec = """
<aiml version="0.1">

<output>
    <object name="bank_run" format="length: 2">
        <string
            name="explanation"
            description="A paragraph about what a bank run is."
            format="length: 200 240"
            on-fail-length="noop"
        />
        <url
            name="follow_up_url"
            description="A web URL where I can read more about bank runs."
            required="true"
            format="valid-url"
            on-fail-valid-url="filter"
        />
    </object>
</output>

<prompt>
Explain what a bank run is in a tweet.

@xml_prefix_prompt

{output_schema}

@json_suffix_prompt_v2_wo_none
</prompt>
</aiml>
"""
```

## Using Guardrails to enforce the `AIML` spec

We write the `AIML` spec to a file and then use it to create a `Guard` object. The `Guard` object is used to wrap the LLM API call and enforce the `AIML` spec on the output of the LLM call.


```python
import tempfile
from rich import print

import guardrails as gd

with tempfile.NamedTemporaryFile(mode="w", suffix=".aiml") as f:
    f.write(aiml_spec)
    f.flush()
    guard = gd.Guard.from_aiml(f.name)
```

We can see that the `Guard` object compiles the `AIML` output specification and adds it to the provided prompt.


```python
print(guard.base_prompt)
```


<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace">
Explain what a bank run is in a tweet.


Given below is XML that describes the information to extract from this document and the tags to extract it into.


<span style="font-weight: bold">&lt;</span><span style="color: #ff00ff; text-decoration-color: #ff00ff; font-weight: bold">output</span><span style="color: #000000; text-decoration-color: #000000">&gt;</span>
<span style="color: #000000; text-decoration-color: #000000">    &lt;object </span><span style="color: #808000; text-decoration-color: #808000">name</span><span style="color: #000000; text-decoration-color: #000000">=</span><span style="color: #008000; text-decoration-color: #008000">"bank_run"</span><span style="color: #000000; text-decoration-color: #000000"> </span><span style="color: #808000; text-decoration-color: #808000">format</span><span style="color: #000000; text-decoration-color: #000000">=</span><span style="color: #008000; text-decoration-color: #008000">"length: 2"</span><span style="color: #000000; text-decoration-color: #000000">&gt;</span>
<span style="color: #000000; text-decoration-color: #000000">        &lt;string </span><span style="color: #808000; text-decoration-color: #808000">name</span><span style="color: #000000; text-decoration-color: #000000">=</span><span style="color: #008000; text-decoration-color: #008000">"explanation"</span><span style="color: #000000; text-decoration-color: #000000"> </span><span style="color: #808000; text-decoration-color: #808000">description</span><span style="color: #000000; text-decoration-color: #000000">=</span><span style="color: #008000; text-decoration-color: #008000">"A paragraph about what a bank run is."</span><span style="color: #000000; text-decoration-color: #000000"> </span><span style="color: #808000; text-decoration-color: #808000">format</span><span style="color: #000000; text-decoration-color: #000000">=</span><span style="color: #008000; text-decoration-color: #008000">"length: 200 240"</span><span style="color: #800080; text-decoration-color: #800080">/</span><span style="color: #000000; text-decoration-color: #000000">&gt;</span>
<span style="color: #000000; text-decoration-color: #000000">        &lt;url </span><span style="color: #808000; text-decoration-color: #808000">name</span><span style="color: #000000; text-decoration-color: #000000">=</span><span style="color: #008000; text-decoration-color: #008000">"follow_up_url"</span><span style="color: #000000; text-decoration-color: #000000"> </span><span style="color: #808000; text-decoration-color: #808000">description</span><span style="color: #000000; text-decoration-color: #000000">=</span><span style="color: #008000; text-decoration-color: #008000">"A web URL where I can read more about bank runs."</span><span style="color: #000000; text-decoration-color: #000000"> </span><span style="color: #808000; text-decoration-color: #808000">required</span><span style="color: #000000; text-decoration-color: #000000">=</span><span style="color: #008000; text-decoration-color: #008000">"true"</span><span style="color: #000000; text-decoration-color: #000000"> </span>
<span style="color: #808000; text-decoration-color: #808000">format</span><span style="color: #000000; text-decoration-color: #000000">=</span><span style="color: #008000; text-decoration-color: #008000">"valid-url"</span><span style="color: #800080; text-decoration-color: #800080">/</span><span style="color: #000000; text-decoration-color: #000000">&gt;</span>
<span style="color: #000000; text-decoration-color: #000000">    &lt;</span><span style="color: #800080; text-decoration-color: #800080">/</span><span style="color: #ff00ff; text-decoration-color: #ff00ff">object</span><span style="color: #000000; text-decoration-color: #000000">&gt;</span>
<span style="color: #000000; text-decoration-color: #000000">&lt;</span><span style="color: #800080; text-decoration-color: #800080">/</span><span style="color: #ff00ff; text-decoration-color: #ff00ff">output</span><span style="font-weight: bold">&gt;</span>




ONLY return a valid JSON object <span style="font-weight: bold">(</span>no other text is necessary<span style="font-weight: bold">)</span>. The JSON MUST conform to the XML format, including 
any types and format requests e.g. requests for lists, objects and specific types. Be correct and concise.

JSON Output:


</pre>



Next, we call the `Guard` object with the LLM API call as the first argument and add any additional arguments to the LLM API call as the remaining arguments.


```python
import openai

# Wrap the OpenAI API call with the `guard` object
raw_llm_output, validated_output = guard(openai.Completion.create, engine="text-davinci-003", max_tokens=1024, temperature=0.3)

# Print the validated output from the LLM
print(validated_output)
```


<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="font-weight: bold">{</span>
    <span style="color: #008000; text-decoration-color: #008000">'bank_run'</span>: <span style="font-weight: bold">{</span>
        <span style="color: #008000; text-decoration-color: #008000">'explanation'</span>: <span style="color: #008000; text-decoration-color: #008000">"A bank run is when a large number of customers withdraw their deposits from a bank due to </span>
<span style="color: #008000; text-decoration-color: #008000">concerns about the bank's solvency."</span>,
        <span style="color: #008000; text-decoration-color: #008000">'follow_up_url'</span>: <span style="color: #008000; text-decoration-color: #008000">'https://www.investopedia.com/terms/b/bankrun.asp'</span>
    <span style="font-weight: bold">}</span>
<span style="font-weight: bold">}</span>
</pre>




```python
print(f'Len of explanation: {len(validated_output["bank_run"]["explanation"])}')
```


<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace">Len of explanation: <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">125</span>
</pre>



As we can see, the `explanation` field didn't meet the quality criteria (length between 200 and 280 characters). However, because of the the `noop` action specified in the `AIML` spec, the `Guard` object returned the output of the LLM API call as is.

Next, we change the `AIML` spec to reask the LLM for a correct `explanation` if its length is incorrect. We do this by creating a new `AIML` spec and creating a new `Guard` object.


```python
aiml_spec = """
<aiml version="0.1">

<output>
    <object name="bank_run" format="length: 2">
        <string
            name="explanation"
            description="A paragraph about what a bank run is."
            format="length: 200 240"
            on-fail-length="reask"
        />
        <url
            name="follow_up_url"
            description="A web URL where I can read more about bank runs."
            required="true"
            format="valid-url"
            on-fail-valid-url="filter"
        />
    </object>
</output>

<prompt>
Explain what a bank run is in a tweet.

@xml_prefix_prompt

{output_schema}

@json_suffix_prompt_v2_wo_none
</prompt>
</aiml>
"""

with tempfile.NamedTemporaryFile(mode="w", suffix=".aiml") as f:
    f.write(aiml_spec)
    f.flush()
    guard_with_reask = gd.Guard.from_aiml(f.name)

raw_llm_output, validated_output = guard_with_reask(openai.Completion.create, engine="text-davinci-003", max_tokens=1024, temperature=0.3)

# Print the validated output from the LLM
print(validated_output)
```


<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="font-weight: bold">{</span>
    <span style="color: #008000; text-decoration-color: #008000">'bank_run'</span>: <span style="font-weight: bold">{</span>
        <span style="color: #008000; text-decoration-color: #008000">'explanation'</span>: <span style="color: #008000; text-decoration-color: #008000">'A bank run is when a large number of people withdraw their deposits from a bank due to </span>
<span style="color: #008000; text-decoration-color: #008000">concerns about its solvency. This can cause a financial crisis if the bank is unable to meet the demand for </span>
<span style="color: #008000; text-decoration-color: #008000">withdrawals.'</span>,
        <span style="color: #008000; text-decoration-color: #008000">'follow_up_url'</span>: <span style="color: #008000; text-decoration-color: #008000">'https://www.investopedia.com/terms/b/bankrun.asp'</span>
    <span style="font-weight: bold">}</span>
<span style="font-weight: bold">}</span>
</pre>




```python
print(f'Len of explanation: {len(validated_output["bank_run"]["explanation"])}')
```


<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace">Len of explanation: <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">207</span>
</pre>

