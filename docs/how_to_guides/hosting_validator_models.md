# Hosting Validator Models

Validation using machine learning models is a highly effective way of detecting LLM hallucinations that are not easily or possibly detected using traditional coding techniques. We’ve selected and tuned some ML models to run different validations. These models are wrapped within model validators. This makes the access pattern straightforward — you can use model-based validators the same as any other [validator](https://www.guardrailsai.com/docs/concepts/validators).

The default access pattern is to use these validators locally for development, but this causes a few issues in production. These models have a large memory footprint, and if they’re run in on-process environments without the correct infrastructure (GPUs, memory, etc.), they can run slowly and suboptimally, adding a huge amount of latency to your application.

In this guide, we will walk through hosting a Guardrails validation model with FastAPI.

## Step 1: Pick a Validator

First, choose a validator that you want to use. You can browse a list of all the Guardrails validators on [Guardrails Hub](https://hub.guardrailsai.com/).

:::note
We suggest picking the “ML” filter on the left side of the page in the “Infrastructure Requirements” section to find validators that use models.
:::

![alt_text](./assets/infrastructure_requirements.png "image_tooltip")

For the purpose of this guide, let’s use the [ToxicLanguage validator](https://hub.guardrailsai.com/validator/guardrails/toxic_language).

## Step 2: Find and Download the Validator Model

The validator model is referenced in the validator implementation, as well as in the README.

![alt_text](./assets/validator_in_readme.png "image_tooltip")

As we can see, the ToxicLanguage validator currently uses the [Detoxify model](https://github.com/unitaryai/detoxify). In order to use this, we need to install the necessary libraries:

```bash
# Install FastAPI server dependencies
pip install fastapi uvicorn

# Option 1 - Install required validator dependencies(General approach - Requires Guardrails)
guardrails hub install hub://guardrails/toxic_language --install-local-models

# Option 2 - Install only required dependencies (Validator Specific - From README)
pip install detoxify torch
```

## Step 3: Wrap the model with FastAPI and run the server

Before implementing the API, we first need to define what the API expects as input and what it will return as output so that the API and server can effectively communicate with each other. 

### The Guardrails Standard Inference Endpoint Schema

Guardrails AI recommends using a similar schema for inputs and outputs across all APIs fronting machine learning models. This approach standardizes the implementation for extracting model inference results out of different models in validators. 

The standard inference request can be found [here](https://github.com/guardrails-ai/guardrails/blob/main/guardrails/validator_base.py#L258). 

The standard input definition looks like this: 

```json
{
  "inputs": [
    {
      "name": "text",
      "shape": [1],
      "data": ["Text we want to check for toxic language"],
      "datatype": "BYTES"
    },
    {
      "name": "threshold",
      "shape": [1],
      "data": [0.5],
      "datatype": "FP32"
    }
  ]
}
```

Here, `“text”` contains the text we want to check for toxic language and `“threshold”` contains the confidence threshold for determining toxicity. If the model predicts a toxic level higher than the threshold, then the text is considered toxic. 

The standard output definition looks like this:

```json
{
  "model_name": "model_name",
  "model_version": "1",
  "outputs": [
    {
      "name": "result",
      "datatype": "BYTES",
      "shape": [2],
      "data": ["toxicity", "threat"]
    }
  ]
}
```

Now it’s time to write our `app.py` to set up the FastAPI server:

```python
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Union
import detoxify
import torch
import json

app = FastAPI()
# Initialize the detoxify model once
device = "cpu"
model = detoxify.Detoxify('unbiased-small', device=torch.device(device))

class InputData(BaseModel):
    name: str
    shape: List[int]
    data: Union[List[str], List[float]]
    datatype: str

class InputRequest(BaseModel):
    inputs: List[InputData]

class OutputData(BaseModel):
    name: str
    datatype: str
    shape: List[int]
    data: List[Union[str, List[str]]]

class OutputResponse(BaseModel):
    modelname: str
    modelversion: str
    outputs: List[OutputData]

@app.post("/validate", response_model=OutputResponse)
async def check_toxicity(input_request: InputRequest):
    text = None
    threshold = None
    
    for inp in input_request.inputs:
        if inp.name == "text":
            text = inp.data[0]
        elif inp.name == "threshold":
            threshold = float(inp.data[0])
    
    if text is None or threshold is None:
        raise HTTPException(status_code=400, detail="Invalid input format")
    results = model.predict(text)
    pred_labels = [label for label, score in results.items() if score > threshold]
    output_data = OutputResponse(
        modelname="unbiased-small",
        modelversion="1",
        outputs=[
            OutputData(
                name="result",
                datatype="BYTES",
                shape=[len(pred_labels)],
                data=[pred_labels]
            )
        ]
    )
    
    print(f"Output data: {output_data}")
    return output_data
# Run the app with uvicorn
# Save this script as app.py and run with: uvicorn app:app --reload
```

Our code for the server is pretty simple – we load in the Detoxify model and create a `validate` endpoint where we do two main things: 

1. Extract the text that we want to validate as well as the threshold that we need to figure out the toxicity of our text 

2. Call the detoxify model on the text and return the output in the format described above 

To run our FastAPI server, we just run `uvicorn app:app –reload` and now our server is running on `https://127.0.0.1:8000`: 

![alt_text](./assets/api_running.png "image_tooltip")

## Step 4: Configure our Guard

Now that we have our server up and running, we have to write a guard that calls the ToxicLanguage validator using our new API.

With Guardrails, this is really simple: 

```python
import guardrails as gd
from guardrails.hub import ToxicLanguage

# Create a Guard class
guard = gd.Guard().use(
    ToxicLanguage(use_local=False, on_fail="exception", validation_endpoint="http://127.0.0.1:8000/validate"),
)
try:
    guard.validate("Stop being such a dumb piece of shit. Why can't you comprehend this?")
except Exception as e:
    print(f"Exception: {e}")
```

We simply initialize a new `Guard` and pass in the `ToxicLanguage` validator with the following parameters:

- **`use_local=False`**: This tells Guardrails to use remote inference instead of running the ML model locally.
- **`on_fail="exception"`**: If toxic language is detected, the guard will throw a `ValidationError` explaining which parts of the text are toxic.
- **`validation_endpoint="http://127.0.0.1:8000/validate"`**: This passes in our FastAPI URL as the validation endpoint to the validator, which then makes a request to this URL to run the Detoxify model on the given text.

When we run the above code and invoke the guard, we see that our guard successfully detected toxic language in our text!

```
Exception: Validation failed for field with errors: The following sentences in your response were found to be toxic:

- Stop being such a dumb piece of shit.
```

## Step 5: Conclusion

You’ve learned how to host the ToxicLanguage model using FastAPI and integrate it directly with Guardrails. This setup can be used for various types of validators and ML models.

### Production Considerations
- Make sure to wrap the application in a WSGI server (like Gunicorn).
- Deploy on GPUs to accelerate model inference and increase performance.

