from pydash import get

llms_and_params = {
  "ai21": {
    "input": ["prompt"],
    "output": ["response.completions.data[0].text"],
    "streaming_output": []
  },
  "anthropic": {
    "input": ["system", "messages"],
    "output": ["message.content"],
    "streaming_output": []
  },
  "aleph-alpha": {
    "input": ["prompt"],
    "output": ["completions[0].completion"],
    "streaming_output": []
  },
  "azure-openai": {
    "input": ["prompt", "messages"],
    "output": [
      "completion.choices[0].message.content", # chat
      "choices[0].text", # completion
      "choices[0].message.function_call.arguments", # function calling
    ],
    "streaming_output": [
      "chunk.choices[0].delta.content", #streaming
    ]
  },
  "cohere": {
    "input": ["chat_history", "message"],
    "output": [
      ""
    ],
    "streaming_output": []
  },
  "gemini": {
    "input": ["messages"],
    "output": [
      "text",
      "chunk.text" # streaming
    ],
    "streaming_output": []
  },
  "ollama": {
    "input": ["prompt", "messages"],
    "output": [
      "response['message']['content']",
    ],
    "streaming_output": [
      "chunk['message']['content']" # streaming
    ]
  },
  "openai": {
    "input": ["prompt", "messages"],
    "output": [
      "completion.choices[0].message.content", # chat
      "openai_response.choices[0].text", # completion
      "openai_response.choices[0].message.function_call.arguments", # function calling
    ],
    "streaming_output": [
      "chunk.choices[0].delta.content", #streaming
    ]
  },
  "together": {
    "input": ["prompt"],
    "output": ["['output']['choices'][0]['text']"],
    "streaming_output": []
  },
}


def extract_text_from_output(response):
  # iterate llms_and_params
  for llm, params in llms_and_params.items():
    for output_path in params["output"]:
      # check if response has that output path in it
      text_check = get(response, output_path)
      if text_check is not None:
        return text_check
  
  raise ValueError("No output path found in response")

def extract_text_from_streaming_output(response):
  # iterate llms_and_params
  for llm, params in llms_and_params.items():
    for output_path in params["streaming_output"]:
      # check if response has that output path in it
      text_check = get(response, output_path)
      if text_check is not None:
        return text_check
  
  raise ValueError("No output path found in response")