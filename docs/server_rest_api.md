---
title: Guardrails API v0.0.0
language_tabs:
  - python: Python
toc_footers: []
includes: []
search: true
highlight_theme: darkula
headingLevel: 2

---
<!-- this was generated with widdershins --environment env.json service-specs/guardrails-service-spec.yml -o api_ref01.md --expandBody true -->
<!-- Generator: Widdershins v4.0.1 -->

<h1 id="guardrails-api">Guardrails Server API</h1>

> A guardrails API server can be iniated via the CLI with `guardrails start`. It can be interacted with via `Guard(`.  This is reference desribes the underlying http interface the client interacts with the server via.

Guardrails CRUD API

# Authentication

- HTTP Authentication, scheme: bearer 

* API Key (ApiKeyAuth)
    - Parameter Name: **authorization**, in: header. 

<h1 id="guardrails-api-service-health">Service Health</h1>

## get__health-check

`GET /health-check`

> Example responses

> 200 Response

```json
{
  "status": 0,
  "message": "string"
}
```

<h3 id="get__health-check-responses">Responses</h3>

|Status|Meaning|Description|Schema|
|---|---|---|---|
|200|[OK](https://tools.ietf.org/html/rfc7231#section-6.3.1)|Returns the status of the server|[HealthCheck](#schemahealthcheck)|
|default|Default|Unexpected error|[HttpError](#schemahttperror)|

<aside class="warning">
To perform this operation, you must be authenticated by means of one of the following methods:
ApiKeyAuth, BearerAuth
</aside>

<h1 id="guardrails-api-guard">guard</h1>

## getGuards

<a id="opIdgetGuards"></a>

`GET /guards`

*Fetches the configuration for all Guards the user has access to.*

> Example responses

> 200 Response

```json
[]
```

<h3 id="getguards-responses">Responses</h3>

|Status|Meaning|Description|Schema|
|---|---|---|---|
|200|[OK](https://tools.ietf.org/html/rfc7231#section-6.3.1)|A list of Guards.|Inline|
|default|Default|Unexpected error|[HttpError](#schemahttperror)|

<h3 id="getguards-responseschema">Response Schema</h3>

Status Code **200**

|Name|Type|Required|Restrictions|Description|
|---|---|---|---|---|
|» *anonymous*|any|false|none|none|

<aside class="warning">
To perform this operation, you must be authenticated by means of one of the following methods:
ApiKeyAuth, BearerAuth
</aside>

## createGuard

<a id="opIdcreateGuard"></a>

`POST /guards`

*Creates a Guard*

> Body parameter

```json
false
```

> Example responses

> default Response

```json
{
  "status": 0,
  "message": "string",
  "cause": "string",
  "fields": {
    "property1": [
      "string"
    ],
    "property2": [
      "string"
    ]
  },
  "context": "string"
}
```

<h3 id="createguard-responses">Responses</h3>

|Status|Meaning|Description|Schema|
|---|---|---|---|
|200|[OK](https://tools.ietf.org/html/rfc7231#section-6.3.1)|the new Guard|None|
|default|Default|Unexpected error|[HttpError](#schemahttperror)|

<h3 id="createguard-responseschema">Response Schema</h3>

<aside class="warning">
To perform this operation, you must be authenticated by means of one of the following methods:
ApiKeyAuth, BearerAuth
</aside>

## getGuard

<a id="opIdgetGuard"></a>

`GET /guards/{guardName}`

*Fetches a specific Guard*

<h3 id="getguard-parameters">Parameters</h3>

|Name|In|Type|Required|Description|
|---|---|---|---|---|
|guardName|path|string|true|Guard name|
|asOf|query|string(date-time)|false|Used to query for data as it existed at this date and time|

> Example responses

> default Response

```json
{
  "status": 0,
  "message": "string",
  "cause": "string",
  "fields": {
    "property1": [
      "string"
    ],
    "property2": [
      "string"
    ]
  },
  "context": "string"
}
```

<h3 id="getguard-responses">Responses</h3>

|Status|Meaning|Description|Schema|
|---|---|---|---|
|200|[OK](https://tools.ietf.org/html/rfc7231#section-6.3.1)|the fetched Guard|None|
|default|Default|Unexpected error|[HttpError](#schemahttperror)|

<h3 id="getguard-responseschema">Response Schema</h3>

<aside class="warning">
To perform this operation, you must be authenticated by means of one of the following methods:
ApiKeyAuth, BearerAuth
</aside>

## updateGuard

<a id="opIdupdateGuard"></a>

`PUT /guards/{guardName}`

*Updates a Guard*

> Body parameter

```json
false
```

<h3 id="updateguard-parameters">Parameters</h3>

|Name|In|Type|Required|Description|
|---|---|---|---|---|
|guardName|path|string|true|Guard name|

> Example responses

> default Response

```json
{
  "status": 0,
  "message": "string",
  "cause": "string",
  "fields": {
    "property1": [
      "string"
    ],
    "property2": [
      "string"
    ]
  },
  "context": "string"
}
```

<h3 id="updateguard-responses">Responses</h3>

|Status|Meaning|Description|Schema|
|---|---|---|---|
|200|[OK](https://tools.ietf.org/html/rfc7231#section-6.3.1)|the updated Guard|None|
|default|Default|Unexpected error|[HttpError](#schemahttperror)|

<h3 id="updateguard-responseschema">Response Schema</h3>

<aside class="warning">
To perform this operation, you must be authenticated by means of one of the following methods:
ApiKeyAuth, BearerAuth
</aside>

## deleteGuard

<a id="opIddeleteGuard"></a>

`DELETE /guards/{guardName}`

*Deletes a Guard*

<h3 id="deleteguard-parameters">Parameters</h3>

|Name|In|Type|Required|Description|
|---|---|---|---|---|
|guardName|path|string|true|Guard name|

> Example responses

> default Response

```json
{
  "status": 0,
  "message": "string",
  "cause": "string",
  "fields": {
    "property1": [
      "string"
    ],
    "property2": [
      "string"
    ]
  },
  "context": "string"
}
```

<h3 id="deleteguard-responses">Responses</h3>

|Status|Meaning|Description|Schema|
|---|---|---|---|
|200|[OK](https://tools.ietf.org/html/rfc7231#section-6.3.1)|the deleted Guard|None|
|default|Default|Unexpected error|[HttpError](#schemahttperror)|

<h3 id="deleteguard-responseschema">Response Schema</h3>

<aside class="warning">
To perform this operation, you must be authenticated by means of one of the following methods:
ApiKeyAuth, BearerAuth
</aside>

## getGuardHistory

<a id="opIdgetGuardHistory"></a>

`GET /guards/{guardName}/history/{callId}`

*Fetches the history for a specific Guard execution by using the id for the most recent Call*

<h3 id="getguardhistory-parameters">Parameters</h3>

|Name|In|Type|Required|Description|
|---|---|---|---|---|
|guardName|path|string|true|Guard name|
|callId|path|string|true|Call id|

> Example responses

> 200 Response

```json
[]
```

<h3 id="getguardhistory-responses">Responses</h3>

|Status|Meaning|Description|Schema|
|---|---|---|---|
|200|[OK](https://tools.ietf.org/html/rfc7231#section-6.3.1)|the fetched Guard History|Inline|
|default|Default|Unexpected error|[HttpError](#schemahttperror)|

<h3 id="getguardhistory-responseschema">Response Schema</h3>

Status Code **200**

|Name|Type|Required|Restrictions|Description|
|---|---|---|---|---|
|» *anonymous*|any|false|none|none|

<aside class="warning">
To perform this operation, you must be authenticated by means of one of the following methods:
ApiKeyAuth, BearerAuth
</aside>

<h1 id="guardrails-api-openai">openai</h1>

## openaiChatCompletion

<a id="opIdopenaiChatCompletion"></a>

`POST /guards/{guardName}/openai/v1/chat/completions`

*OpenAI SDK compatible endpoint for Chat Completions*

> Body parameter

```json
{
  "model": "gpt-3.5-turbo",
  "messages": [
    {
      "role": "string",
      "content": "string"
    }
  ],
  "max_tokens": 0,
  "temperature": 0
}
```

<h3 id="openaichatcompletion-parameters">Parameters</h3>

|Name|In|Type|Required|Description|
|---|---|---|---|---|
|guardName|path|string|true|Guard name|
|body|body|[OpenAIChatCompletionPayload](#schemaopenaichatcompletionpayload)|true|none|
|» model|body|string|false|The model to use for the completion|
|» messages|body|[object]|false|The messages to use for the completion|
|»» role|body|string|false|The role of the message|
|»» content|body|string|false|The content of the message|
|» max_tokens|body|integer|false|The maximum number of tokens to generate|
|» temperature|body|number|false|The sampling temperature|

> Example responses

> 200 Response

```json
{
  "id": "string",
  "created": "string",
  "model_name": "string",
  "choices": [
    {
      "role": "string",
      "content": "string"
    }
  ]
}
```

<h3 id="openaichatcompletion-responses">Responses</h3>

|Status|Meaning|Description|Schema|
|---|---|---|---|
|200|[OK](https://tools.ietf.org/html/rfc7231#section-6.3.1)|The output of the completion|[OpenAIChatCompletion](#schemaopenaichatcompletion)|
|default|Default|Unexpected error|[HttpError](#schemahttperror)|

<aside class="warning">
To perform this operation, you must be authenticated by means of one of the following methods:
ApiKeyAuth, BearerAuth
</aside>

<h1 id="guardrails-api-validate">validate</h1>

## validate

<a id="opIdvalidate"></a>

`POST /guards/{guardName}/validate`

*Runs the validations specified in a Guard*

> Body parameter

```json
{
  "llmOutput": "stubbed llm output",
  "numReasks": 0,
  "promptParams": {
    "property1": null,
    "property2": null
  },
  "llmApi": null,
  "property1": null,
  "property2": null
}
```

<h3 id="validate-parameters">Parameters</h3>

|Name|In|Type|Required|Description|
|---|---|---|---|---|
|guardName|path|string|true|Guard name|
|x-openai-api-key|header|string(password)|false|A valid OpenAI API Key for calling LLM's on your behalf|
|body|body|[ValidatePayload](#schemavalidatepayload)|true|none|
|» **additionalProperties**|body|any|false|none|
|» llmOutput|body|string|false|The LLM output as a string or the input prompts for the LLM|
|» numReasks|body|integer|false|An override for the number of re-asks to perform|
|» promptParams|body|object|false|additional params for llm prompts|
|»» **additionalProperties**|body|any|false|none|
|» llmApi|body|any|false|none|

> Example responses

> default Response

```json
{
  "status": 0,
  "message": "string",
  "cause": "string",
  "fields": {
    "property1": [
      "string"
    ],
    "property2": [
      "string"
    ]
  },
  "context": "string"
}
```

<h3 id="validate-responses">Responses</h3>

|Status|Meaning|Description|Schema|
|---|---|---|---|
|200|[OK](https://tools.ietf.org/html/rfc7231#section-6.3.1)|The output of the validation|None|
|default|Default|Unexpected error|[HttpError](#schemahttperror)|

<h3 id="validate-responseschema">Response Schema</h3>

<aside class="warning">
To perform this operation, you must be authenticated by means of one of the following methods:
ApiKeyAuth, BearerAuth
</aside>

## Schemas

### OpenAIChatCompletionPayload
<!-- backwards compatibility -->
<a id="schemaopenaichatcompletionpayload"></a>
<a id="schema_OpenAIChatCompletionPayload"></a>
<a id="tocSopenaichatcompletionpayload"></a>
<a id="tocsopenaichatcompletionpayload"></a>

```json
{
  "model": "gpt-3.5-turbo",
  "messages": [
    {
      "role": "string",
      "content": "string"
    }
  ],
  "max_tokens": 0,
  "temperature": 0
}

```

#### Properties

|Name|Type|Required|Restrictions|Description|
|---|---|---|---|---|
|model|string|false|none|The model to use for the completion|
|messages|[object]|false|none|The messages to use for the completion|
|» role|string|false|none|The role of the message|
|» content|string|false|none|The content of the message|
|max_tokens|integer|false|none|The maximum number of tokens to generate|
|temperature|number|false|none|The sampling temperature|

### OpenAIChatCompletion
<!-- backwards compatibility -->
<a id="schemaopenaichatcompletion"></a>
<a id="schema_OpenAIChatCompletion"></a>
<a id="tocSopenaichatcompletion"></a>
<a id="tocsopenaichatcompletion"></a>

```json
{
  "id": "string",
  "created": "string",
  "model_name": "string",
  "choices": [
    {
      "role": "string",
      "content": "string"
    }
  ]
}

```

#### Properties

|Name|Type|Required|Restrictions|Description|
|---|---|---|---|---|
|id|string|true|none|The id|
|created|string|true|none|The created date|
|model_name|string|true|none|The model name|
|choices|[object]|true|none|none|
|» role|string|false|none|The role of the message|
|» content|string|false|none|The content of the message|

### HttpError
<!-- backwards compatibility -->
<a id="schemahttperror"></a>
<a id="schema_HttpError"></a>
<a id="tocShttperror"></a>
<a id="tocshttperror"></a>

```json
{
  "status": 0,
  "message": "string",
  "cause": "string",
  "fields": {
    "property1": [
      "string"
    ],
    "property2": [
      "string"
    ]
  },
  "context": "string"
}

```

#### Properties

|Name|Type|Required|Restrictions|Description|
|---|---|---|---|---|
|status|integer|true|none|A valid http status code|
|message|string|true|none|A message explaining the status|
|cause|string|false|none|Used to describe the origin of an error if that original error has meaning to the client.  This field should add specificity to 'message'.|
|fields|object|false|none|Used to identify specific fields in a JSON body that caused the error.  Typically only used for 4xx type responses.  The key should be the json path to the invalid property and the value should be a list of error messages specific to that property.|
|» **additionalProperties**|[string]|false|none|none|
|context|string|false|none|Used to identify what part of the request caused the error for non-JSON payloads.|

### HealthCheck
<!-- backwards compatibility -->
<a id="schemahealthcheck"></a>
<a id="schema_HealthCheck"></a>
<a id="tocShealthcheck"></a>
<a id="tocshealthcheck"></a>

```json
{
  "status": 0,
  "message": "string"
}

```

#### Properties

|Name|Type|Required|Restrictions|Description|
|---|---|---|---|---|
|status|integer|true|none|A valid http status code|
|message|string|true|none|A message explaining the status|

### ValidationOutcome
<!-- backwards compatibility -->
<a id="schemavalidationoutcome"></a>
<a id="schema_ValidationOutcome"></a>
<a id="tocSvalidationoutcome"></a>
<a id="tocsvalidationoutcome"></a>

```json
false

```

#### Properties

*None*

### Guard
<!-- backwards compatibility -->
<a id="schemaguard"></a>
<a id="schema_Guard"></a>
<a id="tocSguard"></a>
<a id="tocsguard"></a>

```json
false

```

#### Properties

*None*

### ValidatePayload
<!-- backwards compatibility -->
<a id="schemavalidatepayload"></a>
<a id="schema_ValidatePayload"></a>
<a id="tocSvalidatepayload"></a>
<a id="tocsvalidatepayload"></a>

```json
{
  "llmOutput": "stubbed llm output",
  "numReasks": 0,
  "promptParams": {
    "property1": null,
    "property2": null
  },
  "llmApi": null,
  "property1": null,
  "property2": null
}

```

#### Properties

|Name|Type|Required|Restrictions|Description|
|---|---|---|---|---|
|**additionalProperties**|any|false|none|none|
|llmOutput|string|false|none|The LLM output as a string or the input prompts for the LLM|
|numReasks|integer|false|none|An override for the number of re-asks to perform|
|promptParams|object|false|none|additional params for llm prompts|
|» **additionalProperties**|any|false|none|none|
|llmApi|[#/$defs/LLMResource](#schema#/$defs/llmresource)|false|none|none|

