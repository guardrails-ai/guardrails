# LlamaIndex

LlamaIndex is an open source data orchestration framework that simplifies integrating private and public data to build new Large Language Models (LLMs). With this integration, you can use Guardrails AI to validate the output of LlamaIndex LLM calls with minimal code changes to your application. 

The sample below walks through setting up a single vector database for Retrieval-Augemnted Generation (RAG) and then querying the index, using Guardrails AI to ensure the answer doesn't contain [Personally Identifiable Information (PII)](https://www.investopedia.com/terms/p/personally-identifiable-information-pii.asp) and doesn't mention competitive products. 

Guardrails AI works with both LlamaIndex's [query engine](https://docs.llamaindex.ai/en/stable/module_guides/deploying/query_engine/) and its [chat engine](https://docs.llamaindex.ai/en/stable/module_guides/deploying/chat_engines/). The query engine is a generic natural language interface for asking questions of data. The chat engine is a higher-level interface that enables a conversation around your data over time, leveraging both the general language capabilities of an LLM and your own private data to generate accurate, up-to-date responses.

## Prerequisites

This document assumes you have set up Guardrails AI. You should also be familiar with foundational Guardrails AI concepts, such as [guards](https://www.guardrailsai.com/docs/concepts/guard) and [validators](https://www.guardrailsai.com/docs/concepts/validators). For more information, see [Quickstart: In-Application](/docs/getting_started/quickstart).

You should be familiar with the basic concepts of RAG. For the basics, see our blog post on [reducing hallucination issues in GenAI apps](https://www.guardrailsai.com/blog/reduce-ai-hallucinations-provenance-guardrails).

This walkthrough downloads files from Guardrails Hub, our public directory of free validators. If you haven't already, create a [Guardrails Hub API key](https://hub.guardrailsai.com/keys) and run `guardrails configure` to set it. For more information on Guardrails Hub, see the [Guardrails Hub documentation](/docs/concepts/hub). 

Unless you specify another LLM, LlamaIndex uses OpenAI for natural language queries as well as to generate vector embeddings. This requires generating and setting an [OpenAI API key](https://platform.openai.com/api-keys), which you can do on Linux using: 

```bash
export OPENAI_API_KEY=KEY_VALUE
```

And in Windows Cmd or Powershell using: 

```powershell
set OPENAI_API_KEY=KEY_VALUE
```

You will need sufficient OpenAI credits to run this sample (between USD $0.01 and $0.03 per run-through). To use another hosted LLM or a locally-running LLM, [see the LlamaIndex documentation](https://docs.llamaindex.ai/en/stable/understanding/using_llms/using_llms/).

## Install LlamaIndex 

Install the LlamaIndex package: 

```bash
pip install llama-index -q
```

Next, install the [Detect PII](https://hub.guardrailsai.com/validator/guardrails/detect_pii) and [Competitor Check](https://hub.guardrailsai.com/validator/guardrails/competitor_check) validators if you don't already have them installed:

```bash
guardrails hub install hub://guardrails/detect_pii --no-install-local-models -q
guardrails hub install hub://guardrails/competitor_check --no-install-local-models -q
```

## Set up your data 

Next, we'll need some sample data to feed into a vector database. [Download the essay located here](https://raw.githubusercontent.com/run-llama/llama_index/main/docs/docs/examples/data/paul_graham/paul_graham_essay.txt) using curl on the command line (or read it with your browser and save it): 

```bash
mkdir -p ./data
curl https://raw.githubusercontent.com/run-llama/llama_index/main/docs/docs/examples/data/paul_graham/paul_graham_essay.txt > ./data/paul_graham_essay.txt
```

This essay from Paul Graham, "What I Worked On," contains details of Graham's life and career, some of which qualify as PII. Additionally, Graham mentions several programming languages - Fortran, Pascal, and BASIC - which, for the purposes of this tutorial, we'll treat as "competitors." 

Next, use the following code [from the LlamaIndex starter tutorial](https://docs.llamaindex.ai/en/stable/getting_started/starter_example/) to create vector embeddings for this document and store the vector database to disk:

```python
import os.path
from llama_index.core import (
    VectorStoreIndex,
    SimpleDirectoryReader,
    StorageContext,
    load_index_from_storage,
)

# check if storage already exists
PERSIST_DIR = "./storage"
if not os.path.exists(PERSIST_DIR):
    # load the documents and create the index
    documents = SimpleDirectoryReader("data").load_data()
    index = VectorStoreIndex.from_documents(documents)
    # store it for later
    index.storage_context.persist(persist_dir=PERSIST_DIR)
else:
    # load the existing index
    storage_context = StorageContext.from_defaults(persist_dir=PERSIST_DIR)
    index = load_index_from_storage(storage_context)
```

By default, this will save the embeddings in a simple document store on disk. You can also use [an open-source or commercial vector database](https://docs.llamaindex.ai/en/stable/module_guides/storing/vector_stores/).

## Validate LlamaIndex calls

Next, call LlamaIndex without any guards to see what values it returns if you don't validate the output. 

```python
query_engine = index.as_query_engine()
response = query_engine.query("What did the author do growing up?")
print(response)
```

You should get back a response like this: 

```bash
The author is Paul Graham. Growing up, he worked on writing short stories and programming, starting with the IBM 1401 in 9th grade using an early version of Fortran. Later, he transitioned to microcomputers like the TRS-80 and began programming more extensively, creating simple games and a word processor.
```

Now, run the same call using the PII and competitor check guards:

```python
from guardrails.integrations.llama_index import GuardrailsQueryEngine

guardrails_query_engine = GuardrailsQueryEngine(engine=query_engine, guard=guard)

response = guardrails_query_engine.query("What did the author do growing up?")
print(response)
```

This replaces the call to LlamaIndex's query engine with the `guardrails.integrations.llama_index.GuardrailsQueryEngine` class, which is a thin wrapper around the LlamaIndex query engine. The response will look something like this: 

```
The author is <PERSON>. Growing up, he worked on writing short stories and programming, starting with the IBM 1401 in 9th grade using an early version of [COMPETITOR]. Later, he transitioned to microcomputers like the TRS-80 and Apple II, where he wrote simple games, programs, and a word processor.
```

To use Guardrails AI validators with LlamaIndex's chat engine, use the `GuardrailsChatEngine` class instead: 

```python
from guardrails.integrations.llama_index import GuardrailsChatEngine
chat_engine = index.as_chat_engine()
guardrails_chat_engine = GuardrailsChatEngine(engine=chat_engine, guard=guard)

response = guardrails_chat_engine.chat("Tell me what the author did growing up.")
print(response)
```