import Tabs from '@theme/Tabs';
import TabItem from '@theme/TabItem';

# Quickstart: Guardrails Server

# Overview

In Guardrails v0.5.0, we released the Guardrails Server. The Guardrails server unlocks several usecases and programming languages through features like **OpenAI SDK compatible enpdoints**, remote validator executions, and server-side support of custom functions.

Together, these features make it easier to get started, and make it possible to host Guardrails in your production infrastructure.

This document will overview a few of the key features of the Guardrails Server, and how to get started.

# Walkthrough

## 0. Configure Guardrails
First, get a free auth key from [Guardrails Hub](https://hub.guardrailsai.com/keys). Then, configure the Guardrails CLI with the auth key.

```bash
guardrails configure
```

## 1. Install the Guardrails Server
This is done by simply installing the `guardrails-ai` package. See the [installation guide](./quickstart.md) for more information.

```bash
pip install guardrails-ai;
guardrails configure;
```

## 2. Create a Guardrails config file
The Guardrails config file is a python file that includes the Guardrails that you want to use, defined in a `Guard` object.

We'll use the `create` command on the guardrails CLI to do this. We'll specify the [GibberishText validator](https://hub.guardrailsai.com/validator/guardrails/gibberish_text) from the Guardrails Hub.


```bash
guardrails create --validators hub://guardrails/gibberish_text --guard-name gibberish_guard
```

:::note
This creates a file called config.py with a Guard object that uses the GibberishText validator. This file can be edited to include more guards, or to change guard behavior.
:::

Update the guard to have the GibberishText validator throw an exception when it is violated. It should look like this

```python
from guardrails import Guard
from guardrails.hub import GibberishText
guard = Guard(name='gibberish_guard')
guard.use(GibberishText(on_fail='exception'))
```

## 3. Start the Guardrails Server
The guardrails CLI starts the server on `localhost:8000` by default. An API reference can be found at `https://localhost:8000/docs`. Since guards run on the backend, you also want to set LLM API Keys in the environment.

```bash
export OPENAI_API_KEY=your_openai_api_key
guardrails start --config config.py
```

Guardrails is now running on localhost:8000.

## 4. Update client to use the Guardrails Server

### OpenAI SDK Integration
You need only route your openai (or openai compatible sdk) base_url to the `http://localhost:8000/guards/[guard_name]/openai/v1/` endpoint. Your client code will now throw an exception if the GibberishText validator is violated. Note, this works in multiple languages.

<Tabs>

<TabItem value="py" label="Python">

```python
from openai import OpenAI

client = OpenAI(
  base_url='http://127.0.0.1:8000/guards/gibberish_guard/openai/v1',
)

response = client.chat.completions.create(
    model="gpt-3.5-turbo",
    messages=[{
        "role": "user",
        "content": "Make up some gibberish for me please!"
    }]
)

print(response.choices[0].message.content)
print(response.guardrails['validation_passed'])
```

</TabItem>
<TabItem value="js" label="JavaScript">

```javascript
const { OpenAI } = require("openai");

const openai = new OpenAI({baseURL: "http://127.0.0.1:8000/guards/gibberish_guard/openai/v1/"});

async function main() {
  const completion = await openai.chat.completions.create({
    messages: [{ role: "system", content: "tell me some gibberish." }],
    model: "gpt-3.5-turbo",
  });

  console.log(completion.choices[0]);
  console.log(completion.guardrails);
}

main();
```

</TabItem>

</Tabs>


:::note
A `guardrails` key is added to the response object, which includes the validation results.
:::

### Advanced Client Usage
Advanced client usage is available in Python. You can point a Guard shim to the Guardrails server and use it as a normal Guard object. Default values can be set in the environment variables `GUARDRAILS_BASE_URL` for the URL and `GUARDRAILS_API_KEY` for the API key.

```python
# Client code
from guardrails import Guard

name_guard = Guard.fetch_guard(name="gibberish_guard", base_url="http://myserver.com", api_key="exampleKey")

validation_outcome = name_guard.validate("floofy doopy boopy")
```

#### Guardrails < v0.6.5
In older versions of Guardrails, you must set the URL and API key through the environment variables mentioned above.

#### Guardrails < v0.5.9
In older versions of Guardrails, you need to set the `use_server` var in settings to True.

```python
# Client code
from guardrails import Guard, settings

settings.use_server = True
name_guard = Guard(name="gibberish_guard")

validation_outcome = name_guard.validate("floofy doopy boopy")
```



## Learn More
- [Guardrails Server Concepts](../concepts/deploying)
- [Guardrails CLI Reference](../cli)
- [Remote Validation Inference](../concepts/remote_validation_inference)
- [Guardrails Hub](https://hub.guardrailsai.com)

