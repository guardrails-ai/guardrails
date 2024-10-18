---
title: Guardrails Server Rest API
language_tabs:
  - shell: Shell
  - http: HTTP
  - javascript: JavaScript
  - ruby: Ruby
  - python: Python
  - php: PHP
  - java: Java
  - go: Go
toc_footers: []
includes: []
search: true
highlight_theme: darkula
headingLevel: 2

---

<!-- Generator: Widdershins v4.0.1 -->

<h1 id="guardrails-api">Guardrails Server REST API</h1>
:::note

Guardrails CRUD API. The server hosts the documenation on this page in an interactive swagger UI. Typically these are available at [localhost:8000/docs](http://localhost:8000/docs) if you are running the server locally with default configurations. 

:::

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
[
  {
    "id": "string",
    "name": "string",
    "description": "string",
    "validators": [
      {
        "id": "string",
        "on": "messages",
        "onFail": "exception",
        "args": [
          true
        ],
        "kwargs": {
          "property1": null,
          "property2": null
        }
      }
    ],
    "output_schema": {
      "definitions": {},
      "dependencies": {
        "property1": {},
        "property2": {}
      },
      "$anchor": "string",
      "$ref": "../dictionary",
      "$dynamicRef": "../dictionary",
      "$dynamicAnchor": "string",
      "$vocabulary": {
        "property1": true,
        "property2": true
      },
      "$comment": "string",
      "$defs": {},
      "prefixItems": [
        null
      ],
      "items": null,
      "contains": null,
      "additionalProperties": null,
      "properties": {},
      "patternProperties": {},
      "dependentSchemas": {
        "property1": null,
        "property2": null
      },
      "propertyNames": null,
      "if": null,
      "then": null,
      "else": null,
      "allOf": [
        null
      ],
      "anyOf": [
        null
      ],
      "oneOf": [
        null
      ],
      "not": null,
      "unevaluatedItems": null,
      "unevaluatedProperties": null,
      "multipleOf": 0,
      "maximum": 0,
      "exclusiveMaximum": 0,
      "minimum": 0,
      "exclusiveMinimum": 0,
      "maxLength": 0,
      "minLength": 0,
      "pattern": "/regex/",
      "maxItems": 0,
      "minItems": 0,
      "uniqueItems": false,
      "maxContains": 0,
      "minContains": 0,
      "maxProperties": 0,
      "minProperties": 0,
      "required": [],
      "dependentRequired": {
        "property1": [],
        "property2": []
      },
      "const": null,
      "enum": [
        null
      ],
      "type": "array",
      "title": "string",
      "description": "string",
      "default": null,
      "deprecated": false,
      "readOnly": false,
      "writeOnly": false,
      "examples": [
        null
      ],
      "format": "string",
      "contentMediaType": "string",
      "contentEncoding": "string",
      "contentSchema": null
    },
    "history": [
      {
        "id": "string",
        "iterations": [],
        "inputs": {
          "llmApi": "string",
          "llmOutput": "string",
          "messages": [
            {
              "property1": null,
              "property2": null
            }
          ],
          "promptParams": {
            "property1": null,
            "property2": null
          },
          "numReasks": 0,
          "metadata": {
            "property1": null,
            "property2": null
          },
          "fullSchemaReask": true,
          "stream": true,
          "args": [
            true
          ],
          "kwargs": {
            "property1": null,
            "property2": null
          }
        },
        "exception": "string"
      }
    ]
  }
]
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
|*anonymous*|[[guard](#schemaguard)]|false|none|none|
|» Guard|[guard](#schemaguard)|false|none|none|
|»» id|string(kebab-case)|true|none|The unique identifier for the Guard.|
|»» name|string|true|none|The name for the Guard.|
|»» description|string|false|none|A description that concisely states the expected behaviour or purpose of the Guard.|
|»» validators|[allOf]|false|none|none|
|»»» ValidatorReference|[ValidatorReference](#schemavalidatorreference)|false|none|none|
|»»»» id|string|true|none|The unique identifier for this Validator.  Often the hub id; e.g. guardrails/regex_match|
|»»»» on|any|false|none|A reference to the property this validator should be applied against.  Can be a valid JSON path or a meta-property such as "messages" or "output"|

*anyOf*

|Name|Type|Required|Restrictions|Description|
|---|---|---|---|---|
|»»»»» *anonymous*|string|false|none|meta-property|

*or*

|Name|Type|Required|Restrictions|Description|
|---|---|---|---|---|
|»»»»» *anonymous*|string|false|none|JSON path to property; e.g. $.foo.bar|

*continued*

|Name|Type|Required|Restrictions|Description|
|---|---|---|---|---|
|»»»» onFail|string|false|none|none|
|»» output_schema|[schema](#schemaschema)|false|none|none|
|»»» definitions|object|false|none|none|
|»»»» **additionalProperties**|any|false|none|none|
|»»» dependencies|object|false|none|none|
|»»»» **additionalProperties**|any|false|none|none|

*anyOf*

|Name|Type|Required|Restrictions|Description|
|---|---|---|---|---|
|»»»»» *anonymous*|any|false|none|none|

*or*

|Name|Type|Required|Restrictions|Description|
|---|---|---|---|---|
|»»»»» *anonymous*|[string]|false|none|none|

*allOf*

|Name|Type|Required|Restrictions|Description|
|---|---|---|---|---|
|»»» *anonymous*|[core](#schemacore)|false|none|none|
|»»»» $anchor|string|false|none|none|
|»»»» $ref|string(uri-reference)|false|none|none|
|»»»» $dynamicRef|string(uri-reference)|false|none|none|
|»»»» $dynamicAnchor|string|false|none|none|
|»»»» $vocabulary|object|false|none|none|
|»»»»» **additionalProperties**|boolean|false|none|none|
|»»»» $comment|string|false|none|none|
|»»»» $defs|object|false|none|none|
|»»»»» **additionalProperties**|any|false|none|none|

*and*

|Name|Type|Required|Restrictions|Description|
|---|---|---|---|---|
|»»» *anonymous*|[applicator](#schemaapplicator)|false|none|none|
|»»»» prefixItems|[any]|false|none|none|
|»»»» items|any|false|none|none|
|»»»» contains|any|false|none|none|
|»»»» additionalProperties|any|false|none|none|
|»»»» properties|object|false|none|none|
|»»»»» **additionalProperties**|any|false|none|none|
|»»»» patternProperties|object|false|none|none|
|»»»»» **additionalProperties**|any|false|none|none|
|»»»» dependentSchemas|object|false|none|none|
|»»»»» **additionalProperties**|any|false|none|none|
|»»»» propertyNames|any|false|none|none|
|»»»» if|any|false|none|none|
|»»»» then|any|false|none|none|
|»»»» else|any|false|none|none|
|»»»» allOf|[any]|false|none|none|
|»»»» anyOf|[any]|false|none|none|
|»»»» oneOf|[any]|false|none|none|
|»»»» not|any|false|none|none|

*and*

|Name|Type|Required|Restrictions|Description|
|---|---|---|---|---|
|»»» *anonymous*|[unevaluated](#schemaunevaluated)|false|none|none|
|»»»» unevaluatedItems|any|false|none|none|
|»»»» unevaluatedProperties|any|false|none|none|

*and*

|Name|Type|Required|Restrictions|Description|
|---|---|---|---|---|
|»»» *anonymous*|[validation](#schemavalidation)|false|none|none|
|»»»» multipleOf|number|false|none|none|
|»»»» maximum|number|false|none|none|
|»»»» exclusiveMaximum|number|false|none|none|
|»»»» minimum|number|false|none|none|
|»»»» exclusiveMinimum|number|false|none|none|
|»»»» maxLength|[nonNegativeInteger](#schemanonnegativeinteger)|false|none|none|
|»»»» minLength|[nonNegativeInteger](#schemanonnegativeinteger)|false|none|none|
|»»»» pattern|string(regex)|false|none|none|
|»»»» maxItems|[nonNegativeInteger](#schemanonnegativeinteger)|false|none|none|
|»»»» minItems|[nonNegativeInteger](#schemanonnegativeinteger)|false|none|none|
|»»»» uniqueItems|boolean|false|none|none|
|»»»» maxContains|[nonNegativeInteger](#schemanonnegativeinteger)|false|none|none|
|»»»» minContains|[nonNegativeInteger](#schemanonnegativeinteger)|false|none|none|
|»»»» maxProperties|[nonNegativeInteger](#schemanonnegativeinteger)|false|none|none|
|»»»» minProperties|[nonNegativeInteger](#schemanonnegativeinteger)|false|none|none|
|»»»» required|[string]|false|none|none|
|»»»» dependentRequired|object|false|none|none|
|»»»»» **additionalProperties**|[string]|false|none|none|
|»»»» const|any|false|none|none|
|»»»» enum|[any]|false|none|none|
|»»»» type|any|false|none|none|

*anyOf*

|Name|Type|Required|Restrictions|Description|
|---|---|---|---|---|
|»»»»» *anonymous*|any|false|none|none|

*or*

|Name|Type|Required|Restrictions|Description|
|---|---|---|---|---|
|»»»»» *anonymous*|[[simpleTypes](#schemasimpletypes)]|false|none|none|

*and*

|Name|Type|Required|Restrictions|Description|
|---|---|---|---|---|
|»»» *anonymous*|[meta-data](#schemameta-data)|false|none|none|
|»»»» title|string|false|none|none|
|»»»» description|string|false|none|none|
|»»»» default|any|false|none|none|
|»»»» deprecated|boolean|false|none|none|
|»»»» readOnly|boolean|false|none|none|
|»»»» writeOnly|boolean|false|none|none|
|»»»» examples|[any]|false|none|none|

*and*

|Name|Type|Required|Restrictions|Description|
|---|---|---|---|---|
|»»» *anonymous*|[format-annotation](#schemaformat-annotation)|false|none|none|
|»»»» format|string|false|none|none|

*and*

|Name|Type|Required|Restrictions|Description|
|---|---|---|---|---|
|»»» *anonymous*|[content](#schemacontent)|false|none|none|
|»»»» contentMediaType|string|false|none|none|
|»»»» contentEncoding|string|false|none|none|
|»»»» contentSchema|any|false|none|none|

*continued*

|Name|Type|Required|Restrictions|Description|
|---|---|---|---|---|
|»» history|[[call](#schemacall)]|false|read-only|none|
|»»» Call|[call](#schemacall)|false|none|none|
|»»»» id|string|true|none|The unique identifier for this Call.  Can be used as an identifier for a specific execution of a Guard.|
|»»»» iterations|[[iteration](#schemaiteration)]|false|none|none|
|»»»»» Iteration|[iteration](#schemaiteration)|false|none|none|
|»»»»»» id|string|true|none|The unique identifier for this Call.  Can be used as an identifier for a specific execution of a Guard.|
|»»»»»» index|integer|true|none|The zero-based index of this iteration within the current Call.|
|»»»»»» callId|string|true|none|The unique identifier for the Call that this iteration is a part of.|
|»»»»»» inputs|[inputs](#schemainputs)|false|none|none|
|»»»»»»» llmApi|string|false|none|The LLM resource targeted by the user. e.g. openai.chat.completions.create|
|»»»»»»» llmOutput|string|false|none|The string output from an external LLM call provided by the user via Guard.parse.|
|»»»»»»» messages|[object]|false|none|The message history for chat models.|
|»»»»»»»» **additionalProperties**|any|false|none|none|
|»»»»»»» promptParams|object|false|none|Parameters to be formatted into the prompt.|
|»»»»»»»» **additionalProperties**|any|false|none|none|
|»»»»»»» numReasks|integer|false|none|The total number of times the LLM can be called to correct output excluding the initial call.|
|»»»»»»» metadata|object|false|none|Additional data to be used by Validators during execution time.|
|»»»»»»»» **additionalProperties**|any|false|none|none|
|»»»»»»» fullSchemaReask|boolean|false|none|Whether to perform reasks for the entire schema rather than for individual fields.|
|»»»»»»» stream|boolean|false|none|Whether to use streaming.|
|»»»»»» outputs|[outputs](#schemaoutputs)|false|none|none|
|»»»»»»» llmResponseInfo|[LLMResponse](#schemallmresponse)|false|none|Information from the LLM response.|
|»»»»»»»» promptTokenCount|integer|false|none|none|
|»»»»»»»» responseTokenCount|integer|false|none|none|
|»»»»»»»» output|string|true|none|none|
|»»»»»»»» streamOutput|[string]|false|none|none|
|»»»»»»»» asyncStreamOutput|[string]|false|none|none|
|»»»»»»» rawOutput|string|false|none|The string content from the LLM response.|
|»»»»»»» parsedOutput|any|false|none|none|

*anyOf*

|Name|Type|Required|Restrictions|Description|
|---|---|---|---|---|
|»»»»»»»» *anonymous*|string|false|none|none|

*or*

|Name|Type|Required|Restrictions|Description|
|---|---|---|---|---|
|»»»»»»»» *anonymous*|object|false|none|none|
|»»»»»»»»» **additionalProperties**|any|false|none|none|

*or*

|Name|Type|Required|Restrictions|Description|
|---|---|---|---|---|
|»»»»»»»» *anonymous*|[anyOf]|false|none|none|
|»»»»»»»»» AnyType|any|false|none|none|

*anyOf*

|Name|Type|Required|Restrictions|Description|
|---|---|---|---|---|
|»»»»»»»»»» *anonymous*|any|false|none|none|

*anyOf*

|Name|Type|Required|Restrictions|Description|
|---|---|---|---|---|
|»»»»»»»»»»» *anonymous*|boolean|false|none|none|

*or*

|Name|Type|Required|Restrictions|Description|
|---|---|---|---|---|
|»»»»»»»»»»» *anonymous*|integer|false|none|none|

*or*

|Name|Type|Required|Restrictions|Description|
|---|---|---|---|---|
|»»»»»»»»»»» *anonymous*|null|false|none|none|

*or*

|Name|Type|Required|Restrictions|Description|
|---|---|---|---|---|
|»»»»»»»»»»» *anonymous*|number|false|none|none|

*or*

|Name|Type|Required|Restrictions|Description|
|---|---|---|---|---|
|»»»»»»»»»»» *anonymous*|string|false|none|none|

*or*

|Name|Type|Required|Restrictions|Description|
|---|---|---|---|---|
|»»»»»»»»»» *anonymous*|[objects](#schemaobjects)|false|none|none|
|»»»»»»»»»»» **additionalProperties**|any|false|none|none|

*or*

|Name|Type|Required|Restrictions|Description|
|---|---|---|---|---|
|»»»»»»»»»» *anonymous*|[anyOf]|false|none|none|

*anyOf*

|Name|Type|Required|Restrictions|Description|
|---|---|---|---|---|
|»»»»»»»»»»» *anonymous*|any|false|none|none|

*or*

|Name|Type|Required|Restrictions|Description|
|---|---|---|---|---|
|»»»»»»»»»»» *anonymous*|[objects](#schemaobjects)|false|none|none|

*continued*

|Name|Type|Required|Restrictions|Description|
|---|---|---|---|---|
|»»»»»»» validationResponse|any|false|none|none|

*anyOf*

|Name|Type|Required|Restrictions|Description|
|---|---|---|---|---|
|»»»»»»»» *anonymous*|string|false|none|none|

*or*

|Name|Type|Required|Restrictions|Description|
|---|---|---|---|---|
|»»»»»»»» *anonymous*|[reask](#schemareask)|false|none|none|
|»»»»»»»»» **additionalProperties**|any|false|none|none|
|»»»»»»»»» incorrectValue|any|false|none|none|
|»»»»»»»»» failResults|[allOf]|false|none|none|
|»»»»»»»»»» FailResult|[fail-result](#schemafail-result)|false|none|The output from a single Validator.|
|»»»»»»»»»»» outcome|any|true|none|none|
|»»»»»»»»»»» errorMessage|string|true|none|none|
|»»»»»»»»»»» fixValue|any|false|none|none|
|»»»»»»»»»»» errorSpans|[[error-span](#schemaerror-span)]|false|none|none|
|»»»»»»»»»»»» ErrorSpan|[error-span](#schemaerror-span)|false|none|none|
|»»»»»»»»»»»»» start|integer|true|none|none|
|»»»»»»»»»»»»» end|integer|true|none|none|
|»»»»»»»»»»»»» reason|string|true|none|The reason validation failed, specific to this chunk.|

*or*

|Name|Type|Required|Restrictions|Description|
|---|---|---|---|---|
|»»»»»»»» *anonymous*|object|false|none|none|
|»»»»»»»»» **additionalProperties**|any|false|none|none|

*or*

|Name|Type|Required|Restrictions|Description|
|---|---|---|---|---|
|»»»»»»»» *anonymous*|[anyOf]|false|none|none|
|»»»»»»»»» AnyType|any|false|none|none|

*continued*

|Name|Type|Required|Restrictions|Description|
|---|---|---|---|---|
|»»»»»»» guardedOutput|any|false|none|none|

*anyOf*

|Name|Type|Required|Restrictions|Description|
|---|---|---|---|---|
|»»»»»»»» *anonymous*|string|false|none|none|

*or*

|Name|Type|Required|Restrictions|Description|
|---|---|---|---|---|
|»»»»»»»» *anonymous*|object|false|none|none|
|»»»»»»»»» **additionalProperties**|any|false|none|none|

*or*

|Name|Type|Required|Restrictions|Description|
|---|---|---|---|---|
|»»»»»»»» *anonymous*|[anyOf]|false|none|none|
|»»»»»»»»» AnyType|any|false|none|none|

*continued*

|Name|Type|Required|Restrictions|Description|
|---|---|---|---|---|
|»»»»»»» reasks|[[reask](#schemareask)]|false|none|none|
|»»»»»»»» ReAsk|[reask](#schemareask)|false|none|none|
|»»»»»»» validatorLogs|[[validator-log](#schemavalidator-log)]|false|none|none|
|»»»»»»»» ValidatorLog|[validator-log](#schemavalidator-log)|false|none|none|
|»»»»»»»»» validatorName|string(PascalCase)|true|none|The class name of the validator.|
|»»»»»»»»» registeredName|string(kebab-case)|true|none|The registry id of the validator.|
|»»»»»»»»» instanceId|any|false|none|A unique identifier for the validator that produced this log within the session.|

*anyOf*

|Name|Type|Required|Restrictions|Description|
|---|---|---|---|---|
|»»»»»»»»»» *anonymous*|string|false|none|none|

*or*

|Name|Type|Required|Restrictions|Description|
|---|---|---|---|---|
|»»»»»»»»»» *anonymous*|integer|false|none|none|

*continued*

|Name|Type|Required|Restrictions|Description|
|---|---|---|---|---|
|»»»»»»»»» propertyPath|string|true|none|The JSON path to the property which was validated that produced this log.|
|»»»»»»»»» valueBeforeValidation|any|true|none|none|
|»»»»»»»»» valueAfterValidation|any|false|none|none|
|»»»»»»»»» validationResult|any|false|none|none|

*anyOf*

|Name|Type|Required|Restrictions|Description|
|---|---|---|---|---|
|»»»»»»»»»» *anonymous*|[pass-result](#schemapass-result)|false|none|The output from a single Validator.|
|»»»»»»»»»»» outcome|any|true|none|none|
|»»»»»»»»»»» valueOverride|any|false|none|none|

*or*

|Name|Type|Required|Restrictions|Description|
|---|---|---|---|---|
|»»»»»»»»»» *anonymous*|[fail-result](#schemafail-result)|false|none|The output from a single Validator.|
|»»»»»»»»»»» outcome|any|true|none|none|
|»»»»»»»»»»» errorMessage|string|true|none|none|
|»»»»»»»»»»» fixValue|any|false|none|none|
|»»»»»»»»»»» errorSpans|[[error-span](#schemaerror-span)]|false|none|none|

*continued*

|Name|Type|Required|Restrictions|Description|
|---|---|---|---|---|
|»»»»»»»»» startTime|string(date-time)|false|none|none|
|»»»»»»»»» endTime|string(date-time)|false|none|none|
|»»»»»»» error|string|false|none|The error message from any exception which interrupted the Guard execution process.|
|»»»» inputs|[CallInputs](#schemacallinputs)|false|none|none|

*allOf*

|Name|Type|Required|Restrictions|Description|
|---|---|---|---|---|
|»»»»» *anonymous*|[inputs](#schemainputs)|false|none|none|

*and*

|Name|Type|Required|Restrictions|Description|
|---|---|---|---|---|
|»»»»» *anonymous*|[args-and-kwargs](#schemaargs-and-kwargs)|false|none|none|
|»»»»»» args|[anyOf]|false|none|none|
|»»»»»»» AnyType|any|false|none|none|
|»»»»»» kwargs|object|false|none|none|
|»»»»»»» **additionalProperties**|any|false|none|none|

*continued*

|Name|Type|Required|Restrictions|Description|
|---|---|---|---|---|
|»»»» exception|string|false|none|none|

#### Enumerated Values

|Property|Value|
|---|---|
|*anonymous*|messages|
|*anonymous*|output|
|onFail|exception|
|onFail|filter|
|onFail|fix|
|onFail|fix_reask|
|onFail|noop|
|onFail|reask|
|onFail|refrain|
|onFail|custom|
|*anonymous*|array|
|*anonymous*|boolean|
|*anonymous*|integer|
|*anonymous*|null|
|*anonymous*|number|
|*anonymous*|object|
|*anonymous*|string|

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
{
  "id": "string",
  "name": "string",
  "description": "string",
  "validators": [
    {
      "id": "string",
      "on": "messages",
      "onFail": "exception",
      "args": [
        true
      ],
      "kwargs": {
        "property1": null,
        "property2": null
      }
    }
  ],
  "output_schema": {
    "definitions": {},
    "dependencies": {
      "property1": {},
      "property2": {}
    },
    "$anchor": "string",
    "$ref": "../dictionary",
    "$dynamicRef": "../dictionary",
    "$dynamicAnchor": "string",
    "$vocabulary": {
      "property1": true,
      "property2": true
    },
    "$comment": "string",
    "$defs": {},
    "prefixItems": [
      null
    ],
    "items": null,
    "contains": null,
    "additionalProperties": null,
    "properties": {},
    "patternProperties": {},
    "dependentSchemas": {
      "property1": null,
      "property2": null
    },
    "propertyNames": null,
    "if": null,
    "then": null,
    "else": null,
    "allOf": [
      null
    ],
    "anyOf": [
      null
    ],
    "oneOf": [
      null
    ],
    "not": null,
    "unevaluatedItems": null,
    "unevaluatedProperties": null,
    "multipleOf": 0,
    "maximum": 0,
    "exclusiveMaximum": 0,
    "minimum": 0,
    "exclusiveMinimum": 0,
    "maxLength": 0,
    "minLength": 0,
    "pattern": "/regex/",
    "maxItems": 0,
    "minItems": 0,
    "uniqueItems": false,
    "maxContains": 0,
    "minContains": 0,
    "maxProperties": 0,
    "minProperties": 0,
    "required": [],
    "dependentRequired": {
      "property1": [],
      "property2": []
    },
    "const": null,
    "enum": [
      null
    ],
    "type": "array",
    "title": "string",
    "description": "string",
    "default": null,
    "deprecated": false,
    "readOnly": false,
    "writeOnly": false,
    "examples": [
      null
    ],
    "format": "string",
    "contentMediaType": "string",
    "contentEncoding": "string",
    "contentSchema": null
  }
}
```

<h3 id="createguard-parameters">Parameters</h3>

|Name|In|Type|Required|Description|
|---|---|---|---|---|
|body|body|[guard](#schemaguard)|true|none|
|» id|body|string(kebab-case)|true|The unique identifier for the Guard.|
|» name|body|string|true|The name for the Guard.|
|» description|body|string|false|A description that concisely states the expected behaviour or purpose of the Guard.|
|» validators|body|[allOf]|false|none|
|»» ValidatorReference|body|[ValidatorReference](#schemavalidatorreference)|false|none|
|»»» id|body|string|true|The unique identifier for this Validator.  Often the hub id; e.g. guardrails/regex_match|
|»»» on|body|any|false|A reference to the property this validator should be applied against.  Can be a valid JSON path or a meta-property such as "prompt" or "output"|
|»»»» *anonymous*|body|string|false|meta-property|
|»»»» *anonymous*|body|string|false|JSON path to property; e.g. $.foo.bar|
|»»» onFail|body|string|false|none|
|» output_schema|body|[schema](#schemaschema)|false|none|
|»» definitions|body|object|false|none|
|»»» **additionalProperties**|body|any|false|none|
|»» dependencies|body|object|false|none|
|»»» **additionalProperties**|body|any|false|none|
|»»»» *anonymous*|body|any|false|none|
|»»»» *anonymous*|body|[string]|false|none|
|»» *anonymous*|body|[core](#schemacore)|false|none|
|»»» $anchor|body|string|false|none|
|»»» $ref|body|string(uri-reference)|false|none|
|»»» $dynamicRef|body|string(uri-reference)|false|none|
|»»» $dynamicAnchor|body|string|false|none|
|»»» $vocabulary|body|object|false|none|
|»»»» **additionalProperties**|body|boolean|false|none|
|»»» $comment|body|string|false|none|
|»»» $defs|body|object|false|none|
|»»»» **additionalProperties**|body|any|false|none|
|»» *anonymous*|body|[applicator](#schemaapplicator)|false|none|
|»»» prefixItems|body|[any]|false|none|
|»»» items|body|any|false|none|
|»»» contains|body|any|false|none|
|»»» additionalProperties|body|any|false|none|
|»»» properties|body|object|false|none|
|»»»» **additionalProperties**|body|any|false|none|
|»»» patternProperties|body|object|false|none|
|»»»» **additionalProperties**|body|any|false|none|
|»»» dependentSchemas|body|object|false|none|
|»»»» **additionalProperties**|body|any|false|none|
|»»» propertyNames|body|any|false|none|
|»»» if|body|any|false|none|
|»»» then|body|any|false|none|
|»»» else|body|any|false|none|
|»»» allOf|body|[any]|false|none|
|»»» anyOf|body|[any]|false|none|
|»»» oneOf|body|[any]|false|none|
|»»» not|body|any|false|none|
|»» *anonymous*|body|[unevaluated](#schemaunevaluated)|false|none|
|»»» unevaluatedItems|body|any|false|none|
|»»» unevaluatedProperties|body|any|false|none|
|»» *anonymous*|body|[validation](#schemavalidation)|false|none|
|»»» multipleOf|body|number|false|none|
|»»» maximum|body|number|false|none|
|»»» exclusiveMaximum|body|number|false|none|
|»»» minimum|body|number|false|none|
|»»» exclusiveMinimum|body|number|false|none|
|»»» maxLength|body|[nonNegativeInteger](#schemanonnegativeinteger)|false|none|
|»»» minLength|body|[nonNegativeInteger](#schemanonnegativeinteger)|false|none|
|»»» pattern|body|string(regex)|false|none|
|»»» maxItems|body|[nonNegativeInteger](#schemanonnegativeinteger)|false|none|
|»»» minItems|body|[nonNegativeInteger](#schemanonnegativeinteger)|false|none|
|»»» uniqueItems|body|boolean|false|none|
|»»» maxContains|body|[nonNegativeInteger](#schemanonnegativeinteger)|false|none|
|»»» minContains|body|[nonNegativeInteger](#schemanonnegativeinteger)|false|none|
|»»» maxProperties|body|[nonNegativeInteger](#schemanonnegativeinteger)|false|none|
|»»» minProperties|body|[nonNegativeInteger](#schemanonnegativeinteger)|false|none|
|»»» required|body|[string]|false|none|
|»»» dependentRequired|body|object|false|none|
|»»»» **additionalProperties**|body|[string]|false|none|
|»»» const|body|any|false|none|
|»»» enum|body|[any]|false|none|
|»»» type|body|any|false|none|
|»»»» *anonymous*|body|any|false|none|
|»»»» *anonymous*|body|[[simpleTypes](#schemasimpletypes)]|false|none|
|»» *anonymous*|body|[meta-data](#schemameta-data)|false|none|
|»»» title|body|string|false|none|
|»»» description|body|string|false|none|
|»»» default|body|any|false|none|
|»»» deprecated|body|boolean|false|none|
|»»» readOnly|body|boolean|false|none|
|»»» writeOnly|body|boolean|false|none|
|»»» examples|body|[any]|false|none|
|»» *anonymous*|body|[format-annotation](#schemaformat-annotation)|false|none|
|»»» format|body|string|false|none|
|»» *anonymous*|body|[content](#schemacontent)|false|none|
|»»» contentMediaType|body|string|false|none|
|»»» contentEncoding|body|string|false|none|
|»»» contentSchema|body|any|false|none|
|» history|body|[[call](#schemacall)]|false|none|
|»» Call|body|[call](#schemacall)|false|none|
|»»» id|body|string|true|The unique identifier for this Call.  Can be used as an identifier for a specific execution of a Guard.|
|»»» iterations|body|[[iteration](#schemaiteration)]|false|none|
|»»»» Iteration|body|[iteration](#schemaiteration)|false|none|
|»»»»» id|body|string|true|The unique identifier for this Call.  Can be used as an identifier for a specific execution of a Guard.|
|»»»»» index|body|integer|true|The zero-based index of this iteration within the current Call.|
|»»»»» callId|body|string|true|The unique identifier for the Call that this iteration is a part of.|
|»»»»» inputs|body|[inputs](#schemainputs)|false|none|
|»»»»»» llmApi|body|string|false|The LLM resource targeted by the user. e.g. openai.chat.completions.create|
|»»»»»» llmOutput|body|string|false|The string output from an external LLM call provided by the user via Guard.parse.|
|»»»»»» messages|body|[object]|false|The message history for chat models.|
|»»»»»»» **additionalProperties**|body|any|false|none|
|»»»»»» promptParams|body|object|false|Parameters to be formatted into the prompt.|
|»»»»»»» **additionalProperties**|body|any|false|none|
|»»»»»» numReasks|body|integer|false|The total number of times the LLM can be called to correct output excluding the initial call.|
|»»»»»» metadata|body|object|false|Additional data to be used by Validators during execution time.|
|»»»»»»» **additionalProperties**|body|any|false|none|
|»»»»»» fullSchemaReask|body|boolean|false|Whether to perform reasks for the entire schema rather than for individual fields.|
|»»»»»» stream|body|boolean|false|Whether to use streaming.|
|»»»»» outputs|body|[outputs](#schemaoutputs)|false|none|
|»»»»»» llmResponseInfo|body|[LLMResponse](#schemallmresponse)|false|Information from the LLM response.|
|»»»»»»» promptTokenCount|body|integer|false|none|
|»»»»»»» responseTokenCount|body|integer|false|none|
|»»»»»»» output|body|string|true|none|
|»»»»»»» streamOutput|body|[string]|false|none|
|»»»»»»» asyncStreamOutput|body|[string]|false|none|
|»»»»»» rawOutput|body|string|false|The string content from the LLM response.|
|»»»»»» parsedOutput|body|any|false|none|
|»»»»»»» *anonymous*|body|string|false|none|
|»»»»»»» *anonymous*|body|object|false|none|
|»»»»»»»» **additionalProperties**|body|any|false|none|
|»»»»»»» *anonymous*|body|[anyOf]|false|none|
|»»»»»»»» AnyType|body|any|false|none|
|»»»»»»»»» *anonymous*|body|any|false|none|
|»»»»»»»»»» *anonymous*|body|boolean|false|none|
|»»»»»»»»»» *anonymous*|body|integer|false|none|
|»»»»»»»»»» *anonymous*|body|null|false|none|
|»»»»»»»»»» *anonymous*|body|number|false|none|
|»»»»»»»»»» *anonymous*|body|string|false|none|
|»»»»»»»»» *anonymous*|body|[objects](#schemaobjects)|false|none|
|»»»»»»»»»» **additionalProperties**|body|any|false|none|
|»»»»»»»»» *anonymous*|body|[anyOf]|false|none|
|»»»»»»»»»» *anonymous*|body|any|false|none|
|»»»»»»»»»» *anonymous*|body|[objects](#schemaobjects)|false|none|
|»»»»»» validationResponse|body|any|false|none|
|»»»»»»» *anonymous*|body|string|false|none|
|»»»»»»» *anonymous*|body|[reask](#schemareask)|false|none|
|»»»»»»»» **additionalProperties**|body|any|false|none|
|»»»»»»»» incorrectValue|body|any|false|none|
|»»»»»»»» failResults|body|[allOf]|false|none|
|»»»»»»»»» FailResult|body|[fail-result](#schemafail-result)|false|The output from a single Validator.|
|»»»»»»»»»» outcome|body|any|true|none|
|»»»»»»»»»» errorMessage|body|string|true|none|
|»»»»»»»»»» fixValue|body|any|false|none|
|»»»»»»»»»» errorSpans|body|[[error-span](#schemaerror-span)]|false|none|
|»»»»»»»»»»» ErrorSpan|body|[error-span](#schemaerror-span)|false|none|
|»»»»»»»»»»»» start|body|integer|true|none|
|»»»»»»»»»»»» end|body|integer|true|none|
|»»»»»»»»»»»» reason|body|string|true|The reason validation failed, specific to this chunk.|
|»»»»»»» *anonymous*|body|object|false|none|
|»»»»»»»» **additionalProperties**|body|any|false|none|
|»»»»»»» *anonymous*|body|[anyOf]|false|none|
|»»»»»»»» AnyType|body|any|false|none|
|»»»»»» guardedOutput|body|any|false|none|
|»»»»»»» *anonymous*|body|string|false|none|
|»»»»»»» *anonymous*|body|object|false|none|
|»»»»»»»» **additionalProperties**|body|any|false|none|
|»»»»»»» *anonymous*|body|[anyOf]|false|none|
|»»»»»»»» AnyType|body|any|false|none|
|»»»»»» reasks|body|[[reask](#schemareask)]|false|none|
|»»»»»»» ReAsk|body|[reask](#schemareask)|false|none|
|»»»»»» validatorLogs|body|[[validator-log](#schemavalidator-log)]|false|none|
|»»»»»»» ValidatorLog|body|[validator-log](#schemavalidator-log)|false|none|
|»»»»»»»» validatorName|body|string(PascalCase)|true|The class name of the validator.|
|»»»»»»»» registeredName|body|string(kebab-case)|true|The registry id of the validator.|
|»»»»»»»» instanceId|body|any|false|A unique identifier for the validator that produced this log within the session.|
|»»»»»»»»» *anonymous*|body|string|false|none|
|»»»»»»»»» *anonymous*|body|integer|false|none|
|»»»»»»»» propertyPath|body|string|true|The JSON path to the property which was validated that produced this log.|
|»»»»»»»» valueBeforeValidation|body|any|true|none|
|»»»»»»»» valueAfterValidation|body|any|false|none|
|»»»»»»»» validationResult|body|any|false|none|
|»»»»»»»»» *anonymous*|body|[pass-result](#schemapass-result)|false|The output from a single Validator.|
|»»»»»»»»»» outcome|body|any|true|none|
|»»»»»»»»»» valueOverride|body|any|false|none|
|»»»»»»»»» *anonymous*|body|[fail-result](#schemafail-result)|false|The output from a single Validator.|
|»»»»»»»»»» outcome|body|any|true|none|
|»»»»»»»»»» errorMessage|body|string|true|none|
|»»»»»»»»»» fixValue|body|any|false|none|
|»»»»»»»»»» errorSpans|body|[[error-span](#schemaerror-span)]|false|none|
|»»»»»»»» startTime|body|string(date-time)|false|none|
|»»»»»»»» endTime|body|string(date-time)|false|none|
|»»»»»» error|body|string|false|The error message from any exception which interrupted the Guard execution process.|
|»»» inputs|body|[CallInputs](#schemacallinputs)|false|none|
|»»»» *anonymous*|body|[inputs](#schemainputs)|false|none|
|»»»» *anonymous*|body|[args-and-kwargs](#schemaargs-and-kwargs)|false|none|
|»»»»» args|body|[anyOf]|false|none|
|»»»»»» AnyType|body|any|false|none|
|»»»»» kwargs|body|object|false|none|
|»»»»»» **additionalProperties**|body|any|false|none|
|»»» exception|body|string|false|none|

#### Enumerated Values

|Parameter|Value|
|---|---|
|»»»» *anonymous*|messages|
|»»»» *anonymous*|output|
|»»» onFail|exception|
|»»» onFail|filter|
|»»» onFail|fix|
|»»» onFail|fix_reask|
|»»» onFail|noop|
|»»» onFail|reask|
|»»» onFail|refrain|
|»»» onFail|custom|
|»»»» *anonymous*|array|
|»»»» *anonymous*|boolean|
|»»»» *anonymous*|integer|
|»»»» *anonymous*|null|
|»»»» *anonymous*|number|
|»»»» *anonymous*|object|
|»»»» *anonymous*|string|
|»»»» *anonymous*|array|
|»»»» *anonymous*|boolean|
|»»»» *anonymous*|integer|
|»»»» *anonymous*|null|
|»»»» *anonymous*|number|
|»»»» *anonymous*|object|
|»»»» *anonymous*|string|

> Example responses

> 200 Response

```json
{
  "id": "string",
  "name": "string",
  "description": "string",
  "validators": [
    {
      "id": "string",
      "on": "messages",
      "onFail": "exception",
      "args": [
        true
      ],
      "kwargs": {
        "property1": null,
        "property2": null
      }
    }
  ],
  "output_schema": {
    "definitions": {},
    "dependencies": {
      "property1": {},
      "property2": {}
    },
    "$anchor": "string",
    "$ref": "../dictionary",
    "$dynamicRef": "../dictionary",
    "$dynamicAnchor": "string",
    "$vocabulary": {
      "property1": true,
      "property2": true
    },
    "$comment": "string",
    "$defs": {},
    "prefixItems": [
      null
    ],
    "items": null,
    "contains": null,
    "additionalProperties": null,
    "properties": {},
    "patternProperties": {},
    "dependentSchemas": {
      "property1": null,
      "property2": null
    },
    "propertyNames": null,
    "if": null,
    "then": null,
    "else": null,
    "allOf": [
      null
    ],
    "anyOf": [
      null
    ],
    "oneOf": [
      null
    ],
    "not": null,
    "unevaluatedItems": null,
    "unevaluatedProperties": null,
    "multipleOf": 0,
    "maximum": 0,
    "exclusiveMaximum": 0,
    "minimum": 0,
    "exclusiveMinimum": 0,
    "maxLength": 0,
    "minLength": 0,
    "pattern": "/regex/",
    "maxItems": 0,
    "minItems": 0,
    "uniqueItems": false,
    "maxContains": 0,
    "minContains": 0,
    "maxProperties": 0,
    "minProperties": 0,
    "required": [],
    "dependentRequired": {
      "property1": [],
      "property2": []
    },
    "const": null,
    "enum": [
      null
    ],
    "type": "array",
    "title": "string",
    "description": "string",
    "default": null,
    "deprecated": false,
    "readOnly": false,
    "writeOnly": false,
    "examples": [
      null
    ],
    "format": "string",
    "contentMediaType": "string",
    "contentEncoding": "string",
    "contentSchema": null
  },
  "history": [
    {
      "id": "string",
      "iterations": [],
      "inputs": {
        "llmApi": "string",
        "llmOutput": "string",
        "messages": [
          {
            "property1": null,
            "property2": null
          }
        ],
        "promptParams": {
          "property1": null,
          "property2": null
        },
        "numReasks": 0,
        "metadata": {
          "property1": null,
          "property2": null
        },
        "fullSchemaReask": true,
        "stream": true,
        "args": [
          true
        ],
        "kwargs": {
          "property1": null,
          "property2": null
        }
      },
      "exception": "string"
    }
  ]
}
```

<h3 id="createguard-responses">Responses</h3>

|Status|Meaning|Description|Schema|
|---|---|---|---|
|200|[OK](https://tools.ietf.org/html/rfc7231#section-6.3.1)|the new Guard|[guard](#schemaguard)|
|default|Default|Unexpected error|[HttpError](#schemahttperror)|

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

> 200 Response

```json
{
  "id": "string",
  "name": "string",
  "description": "string",
  "validators": [
    {
      "id": "string",
      "on": "messages",
      "onFail": "exception",
      "args": [
        true
      ],
      "kwargs": {
        "property1": null,
        "property2": null
      }
    }
  ],
  "output_schema": {
    "definitions": {},
    "dependencies": {
      "property1": {},
      "property2": {}
    },
    "$anchor": "string",
    "$ref": "../dictionary",
    "$dynamicRef": "../dictionary",
    "$dynamicAnchor": "string",
    "$vocabulary": {
      "property1": true,
      "property2": true
    },
    "$comment": "string",
    "$defs": {},
    "prefixItems": [
      null
    ],
    "items": null,
    "contains": null,
    "additionalProperties": null,
    "properties": {},
    "patternProperties": {},
    "dependentSchemas": {
      "property1": null,
      "property2": null
    },
    "propertyNames": null,
    "if": null,
    "then": null,
    "else": null,
    "allOf": [
      null
    ],
    "anyOf": [
      null
    ],
    "oneOf": [
      null
    ],
    "not": null,
    "unevaluatedItems": null,
    "unevaluatedProperties": null,
    "multipleOf": 0,
    "maximum": 0,
    "exclusiveMaximum": 0,
    "minimum": 0,
    "exclusiveMinimum": 0,
    "maxLength": 0,
    "minLength": 0,
    "pattern": "/regex/",
    "maxItems": 0,
    "minItems": 0,
    "uniqueItems": false,
    "maxContains": 0,
    "minContains": 0,
    "maxProperties": 0,
    "minProperties": 0,
    "required": [],
    "dependentRequired": {
      "property1": [],
      "property2": []
    },
    "const": null,
    "enum": [
      null
    ],
    "type": "array",
    "title": "string",
    "description": "string",
    "default": null,
    "deprecated": false,
    "readOnly": false,
    "writeOnly": false,
    "examples": [
      null
    ],
    "format": "string",
    "contentMediaType": "string",
    "contentEncoding": "string",
    "contentSchema": null
  },
  "history": [
    {
      "id": "string",
      "iterations": [],
      "inputs": {
        "llmApi": "string",
        "llmOutput": "string",
        "messages": [
          {
            "property1": null,
            "property2": null
          }
        ],
        "promptParams": {
          "property1": null,
          "property2": null
        },
        "numReasks": 0,
        "metadata": {
          "property1": null,
          "property2": null
        },
        "fullSchemaReask": true,
        "stream": true,
        "args": [
          true
        ],
        "kwargs": {
          "property1": null,
          "property2": null
        }
      },
      "exception": "string"
    }
  ]
}
```

<h3 id="getguard-responses">Responses</h3>

|Status|Meaning|Description|Schema|
|---|---|---|---|
|200|[OK](https://tools.ietf.org/html/rfc7231#section-6.3.1)|the fetched Guard|[guard](#schemaguard)|
|default|Default|Unexpected error|[HttpError](#schemahttperror)|

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
{
  "id": "string",
  "name": "string",
  "description": "string",
  "validators": [
    {
      "id": "string",
      "on": "messages",
      "onFail": "exception",
      "args": [
        true
      ],
      "kwargs": {
        "property1": null,
        "property2": null
      }
    }
  ],
  "output_schema": {
    "definitions": {},
    "dependencies": {
      "property1": {},
      "property2": {}
    },
    "$anchor": "string",
    "$ref": "../dictionary",
    "$dynamicRef": "../dictionary",
    "$dynamicAnchor": "string",
    "$vocabulary": {
      "property1": true,
      "property2": true
    },
    "$comment": "string",
    "$defs": {},
    "prefixItems": [
      null
    ],
    "items": null,
    "contains": null,
    "additionalProperties": null,
    "properties": {},
    "patternProperties": {},
    "dependentSchemas": {
      "property1": null,
      "property2": null
    },
    "propertyNames": null,
    "if": null,
    "then": null,
    "else": null,
    "allOf": [
      null
    ],
    "anyOf": [
      null
    ],
    "oneOf": [
      null
    ],
    "not": null,
    "unevaluatedItems": null,
    "unevaluatedProperties": null,
    "multipleOf": 0,
    "maximum": 0,
    "exclusiveMaximum": 0,
    "minimum": 0,
    "exclusiveMinimum": 0,
    "maxLength": 0,
    "minLength": 0,
    "pattern": "/regex/",
    "maxItems": 0,
    "minItems": 0,
    "uniqueItems": false,
    "maxContains": 0,
    "minContains": 0,
    "maxProperties": 0,
    "minProperties": 0,
    "required": [],
    "dependentRequired": {
      "property1": [],
      "property2": []
    },
    "const": null,
    "enum": [
      null
    ],
    "type": "array",
    "title": "string",
    "description": "string",
    "default": null,
    "deprecated": false,
    "readOnly": false,
    "writeOnly": false,
    "examples": [
      null
    ],
    "format": "string",
    "contentMediaType": "string",
    "contentEncoding": "string",
    "contentSchema": null
  }
}
```

<h3 id="updateguard-parameters">Parameters</h3>

|Name|In|Type|Required|Description|
|---|---|---|---|---|
|guardName|path|string|true|Guard name|
|body|body|[guard](#schemaguard)|true|none|
|» id|body|string(kebab-case)|true|The unique identifier for the Guard.|
|» name|body|string|true|The name for the Guard.|
|» description|body|string|false|A description that concisely states the expected behaviour or purpose of the Guard.|
|» validators|body|[allOf]|false|none|
|»» ValidatorReference|body|[ValidatorReference](#schemavalidatorreference)|false|none|
|»»» id|body|string|true|The unique identifier for this Validator.  Often the hub id; e.g. guardrails/regex_match|
|»»» on|body|any|false|A reference to the property this validator should be applied against.  Can be a valid JSON path or a meta-property such as "prompt" or "output"|
|»»»» *anonymous*|body|string|false|meta-property|
|»»»» *anonymous*|body|string|false|JSON path to property; e.g. $.foo.bar|
|»»» onFail|body|string|false|none|
|» output_schema|body|[schema](#schemaschema)|false|none|
|»» definitions|body|object|false|none|
|»»» **additionalProperties**|body|any|false|none|
|»» dependencies|body|object|false|none|
|»»» **additionalProperties**|body|any|false|none|
|»»»» *anonymous*|body|any|false|none|
|»»»» *anonymous*|body|[string]|false|none|
|»» *anonymous*|body|[core](#schemacore)|false|none|
|»»» $anchor|body|string|false|none|
|»»» $ref|body|string(uri-reference)|false|none|
|»»» $dynamicRef|body|string(uri-reference)|false|none|
|»»» $dynamicAnchor|body|string|false|none|
|»»» $vocabulary|body|object|false|none|
|»»»» **additionalProperties**|body|boolean|false|none|
|»»» $comment|body|string|false|none|
|»»» $defs|body|object|false|none|
|»»»» **additionalProperties**|body|any|false|none|
|»» *anonymous*|body|[applicator](#schemaapplicator)|false|none|
|»»» prefixItems|body|[any]|false|none|
|»»» items|body|any|false|none|
|»»» contains|body|any|false|none|
|»»» additionalProperties|body|any|false|none|
|»»» properties|body|object|false|none|
|»»»» **additionalProperties**|body|any|false|none|
|»»» patternProperties|body|object|false|none|
|»»»» **additionalProperties**|body|any|false|none|
|»»» dependentSchemas|body|object|false|none|
|»»»» **additionalProperties**|body|any|false|none|
|»»» propertyNames|body|any|false|none|
|»»» if|body|any|false|none|
|»»» then|body|any|false|none|
|»»» else|body|any|false|none|
|»»» allOf|body|[any]|false|none|
|»»» anyOf|body|[any]|false|none|
|»»» oneOf|body|[any]|false|none|
|»»» not|body|any|false|none|
|»» *anonymous*|body|[unevaluated](#schemaunevaluated)|false|none|
|»»» unevaluatedItems|body|any|false|none|
|»»» unevaluatedProperties|body|any|false|none|
|»» *anonymous*|body|[validation](#schemavalidation)|false|none|
|»»» multipleOf|body|number|false|none|
|»»» maximum|body|number|false|none|
|»»» exclusiveMaximum|body|number|false|none|
|»»» minimum|body|number|false|none|
|»»» exclusiveMinimum|body|number|false|none|
|»»» maxLength|body|[nonNegativeInteger](#schemanonnegativeinteger)|false|none|
|»»» minLength|body|[nonNegativeInteger](#schemanonnegativeinteger)|false|none|
|»»» pattern|body|string(regex)|false|none|
|»»» maxItems|body|[nonNegativeInteger](#schemanonnegativeinteger)|false|none|
|»»» minItems|body|[nonNegativeInteger](#schemanonnegativeinteger)|false|none|
|»»» uniqueItems|body|boolean|false|none|
|»»» maxContains|body|[nonNegativeInteger](#schemanonnegativeinteger)|false|none|
|»»» minContains|body|[nonNegativeInteger](#schemanonnegativeinteger)|false|none|
|»»» maxProperties|body|[nonNegativeInteger](#schemanonnegativeinteger)|false|none|
|»»» minProperties|body|[nonNegativeInteger](#schemanonnegativeinteger)|false|none|
|»»» required|body|[string]|false|none|
|»»» dependentRequired|body|object|false|none|
|»»»» **additionalProperties**|body|[string]|false|none|
|»»» const|body|any|false|none|
|»»» enum|body|[any]|false|none|
|»»» type|body|any|false|none|
|»»»» *anonymous*|body|any|false|none|
|»»»» *anonymous*|body|[[simpleTypes](#schemasimpletypes)]|false|none|
|»» *anonymous*|body|[meta-data](#schemameta-data)|false|none|
|»»» title|body|string|false|none|
|»»» description|body|string|false|none|
|»»» default|body|any|false|none|
|»»» deprecated|body|boolean|false|none|
|»»» readOnly|body|boolean|false|none|
|»»» writeOnly|body|boolean|false|none|
|»»» examples|body|[any]|false|none|
|»» *anonymous*|body|[format-annotation](#schemaformat-annotation)|false|none|
|»»» format|body|string|false|none|
|»» *anonymous*|body|[content](#schemacontent)|false|none|
|»»» contentMediaType|body|string|false|none|
|»»» contentEncoding|body|string|false|none|
|»»» contentSchema|body|any|false|none|
|» history|body|[[call](#schemacall)]|false|none|
|»» Call|body|[call](#schemacall)|false|none|
|»»» id|body|string|true|The unique identifier for this Call.  Can be used as an identifier for a specific execution of a Guard.|
|»»» iterations|body|[[iteration](#schemaiteration)]|false|none|
|»»»» Iteration|body|[iteration](#schemaiteration)|false|none|
|»»»»» id|body|string|true|The unique identifier for this Call.  Can be used as an identifier for a specific execution of a Guard.|
|»»»»» index|body|integer|true|The zero-based index of this iteration within the current Call.|
|»»»»» callId|body|string|true|The unique identifier for the Call that this iteration is a part of.|
|»»»»» inputs|body|[inputs](#schemainputs)|false|none|
|»»»»»» llmApi|body|string|false|The LLM resource targeted by the user. e.g. openai.chat.completions.create|
|»»»»»» llmOutput|body|string|false|The string output from an external LLM call provided by the user via Guard.parse.|
|»»»»»» messages|body|[object]|false|The message history for chat models.|
|»»»»»»» **additionalProperties**|body|any|false|none|
|»»»»»» promptParams|body|object|false|Parameters to be formatted into the prompt.|
|»»»»»»» **additionalProperties**|body|any|false|none|
|»»»»»» numReasks|body|integer|false|The total number of times the LLM can be called to correct output excluding the initial call.|
|»»»»»» metadata|body|object|false|Additional data to be used by Validators during execution time.|
|»»»»»»» **additionalProperties**|body|any|false|none|
|»»»»»» fullSchemaReask|body|boolean|false|Whether to perform reasks for the entire schema rather than for individual fields.|
|»»»»»» stream|body|boolean|false|Whether to use streaming.|
|»»»»» outputs|body|[outputs](#schemaoutputs)|false|none|
|»»»»»» llmResponseInfo|body|[LLMResponse](#schemallmresponse)|false|Information from the LLM response.|
|»»»»»»» promptTokenCount|body|integer|false|none|
|»»»»»»» responseTokenCount|body|integer|false|none|
|»»»»»»» output|body|string|true|none|
|»»»»»»» streamOutput|body|[string]|false|none|
|»»»»»»» asyncStreamOutput|body|[string]|false|none|
|»»»»»» rawOutput|body|string|false|The string content from the LLM response.|
|»»»»»» parsedOutput|body|any|false|none|
|»»»»»»» *anonymous*|body|string|false|none|
|»»»»»»» *anonymous*|body|object|false|none|
|»»»»»»»» **additionalProperties**|body|any|false|none|
|»»»»»»» *anonymous*|body|[anyOf]|false|none|
|»»»»»»»» AnyType|body|any|false|none|
|»»»»»»»»» *anonymous*|body|any|false|none|
|»»»»»»»»»» *anonymous*|body|boolean|false|none|
|»»»»»»»»»» *anonymous*|body|integer|false|none|
|»»»»»»»»»» *anonymous*|body|null|false|none|
|»»»»»»»»»» *anonymous*|body|number|false|none|
|»»»»»»»»»» *anonymous*|body|string|false|none|
|»»»»»»»»» *anonymous*|body|[objects](#schemaobjects)|false|none|
|»»»»»»»»»» **additionalProperties**|body|any|false|none|
|»»»»»»»»» *anonymous*|body|[anyOf]|false|none|
|»»»»»»»»»» *anonymous*|body|any|false|none|
|»»»»»»»»»» *anonymous*|body|[objects](#schemaobjects)|false|none|
|»»»»»» validationResponse|body|any|false|none|
|»»»»»»» *anonymous*|body|string|false|none|
|»»»»»»» *anonymous*|body|[reask](#schemareask)|false|none|
|»»»»»»»» **additionalProperties**|body|any|false|none|
|»»»»»»»» incorrectValue|body|any|false|none|
|»»»»»»»» failResults|body|[allOf]|false|none|
|»»»»»»»»» FailResult|body|[fail-result](#schemafail-result)|false|The output from a single Validator.|
|»»»»»»»»»» outcome|body|any|true|none|
|»»»»»»»»»» errorMessage|body|string|true|none|
|»»»»»»»»»» fixValue|body|any|false|none|
|»»»»»»»»»» errorSpans|body|[[error-span](#schemaerror-span)]|false|none|
|»»»»»»»»»»» ErrorSpan|body|[error-span](#schemaerror-span)|false|none|
|»»»»»»»»»»»» start|body|integer|true|none|
|»»»»»»»»»»»» end|body|integer|true|none|
|»»»»»»»»»»»» reason|body|string|true|The reason validation failed, specific to this chunk.|
|»»»»»»» *anonymous*|body|object|false|none|
|»»»»»»»» **additionalProperties**|body|any|false|none|
|»»»»»»» *anonymous*|body|[anyOf]|false|none|
|»»»»»»»» AnyType|body|any|false|none|
|»»»»»» guardedOutput|body|any|false|none|
|»»»»»»» *anonymous*|body|string|false|none|
|»»»»»»» *anonymous*|body|object|false|none|
|»»»»»»»» **additionalProperties**|body|any|false|none|
|»»»»»»» *anonymous*|body|[anyOf]|false|none|
|»»»»»»»» AnyType|body|any|false|none|
|»»»»»» reasks|body|[[reask](#schemareask)]|false|none|
|»»»»»»» ReAsk|body|[reask](#schemareask)|false|none|
|»»»»»» validatorLogs|body|[[validator-log](#schemavalidator-log)]|false|none|
|»»»»»»» ValidatorLog|body|[validator-log](#schemavalidator-log)|false|none|
|»»»»»»»» validatorName|body|string(PascalCase)|true|The class name of the validator.|
|»»»»»»»» registeredName|body|string(kebab-case)|true|The registry id of the validator.|
|»»»»»»»» instanceId|body|any|false|A unique identifier for the validator that produced this log within the session.|
|»»»»»»»»» *anonymous*|body|string|false|none|
|»»»»»»»»» *anonymous*|body|integer|false|none|
|»»»»»»»» propertyPath|body|string|true|The JSON path to the property which was validated that produced this log.|
|»»»»»»»» valueBeforeValidation|body|any|true|none|
|»»»»»»»» valueAfterValidation|body|any|false|none|
|»»»»»»»» validationResult|body|any|false|none|
|»»»»»»»»» *anonymous*|body|[pass-result](#schemapass-result)|false|The output from a single Validator.|
|»»»»»»»»»» outcome|body|any|true|none|
|»»»»»»»»»» valueOverride|body|any|false|none|
|»»»»»»»»» *anonymous*|body|[fail-result](#schemafail-result)|false|The output from a single Validator.|
|»»»»»»»»»» outcome|body|any|true|none|
|»»»»»»»»»» errorMessage|body|string|true|none|
|»»»»»»»»»» fixValue|body|any|false|none|
|»»»»»»»»»» errorSpans|body|[[error-span](#schemaerror-span)]|false|none|
|»»»»»»»» startTime|body|string(date-time)|false|none|
|»»»»»»»» endTime|body|string(date-time)|false|none|
|»»»»»» error|body|string|false|The error message from any exception which interrupted the Guard execution process.|
|»»» inputs|body|[CallInputs](#schemacallinputs)|false|none|
|»»»» *anonymous*|body|[inputs](#schemainputs)|false|none|
|»»»» *anonymous*|body|[args-and-kwargs](#schemaargs-and-kwargs)|false|none|
|»»»»» args|body|[anyOf]|false|none|
|»»»»»» AnyType|body|any|false|none|
|»»»»» kwargs|body|object|false|none|
|»»»»»» **additionalProperties**|body|any|false|none|
|»»» exception|body|string|false|none|

#### Enumerated Values

|Parameter|Value|
|---|---|
|»»»» *anonymous*|messages|
|»»»» *anonymous*|output|
|»»» onFail|exception|
|»»» onFail|filter|
|»»» onFail|fix|
|»»» onFail|fix_reask|
|»»» onFail|noop|
|»»» onFail|reask|
|»»» onFail|refrain|
|»»» onFail|custom|
|»»»» *anonymous*|array|
|»»»» *anonymous*|boolean|
|»»»» *anonymous*|integer|
|»»»» *anonymous*|null|
|»»»» *anonymous*|number|
|»»»» *anonymous*|object|
|»»»» *anonymous*|string|
|»»»» *anonymous*|array|
|»»»» *anonymous*|boolean|
|»»»» *anonymous*|integer|
|»»»» *anonymous*|null|
|»»»» *anonymous*|number|
|»»»» *anonymous*|object|
|»»»» *anonymous*|string|

> Example responses

> 200 Response

```json
{
  "id": "string",
  "name": "string",
  "description": "string",
  "validators": [
    {
      "id": "string",
      "on": "messages",
      "onFail": "exception",
      "args": [
        true
      ],
      "kwargs": {
        "property1": null,
        "property2": null
      }
    }
  ],
  "output_schema": {
    "definitions": {},
    "dependencies": {
      "property1": {},
      "property2": {}
    },
    "$anchor": "string",
    "$ref": "../dictionary",
    "$dynamicRef": "../dictionary",
    "$dynamicAnchor": "string",
    "$vocabulary": {
      "property1": true,
      "property2": true
    },
    "$comment": "string",
    "$defs": {},
    "prefixItems": [
      null
    ],
    "items": null,
    "contains": null,
    "additionalProperties": null,
    "properties": {},
    "patternProperties": {},
    "dependentSchemas": {
      "property1": null,
      "property2": null
    },
    "propertyNames": null,
    "if": null,
    "then": null,
    "else": null,
    "allOf": [
      null
    ],
    "anyOf": [
      null
    ],
    "oneOf": [
      null
    ],
    "not": null,
    "unevaluatedItems": null,
    "unevaluatedProperties": null,
    "multipleOf": 0,
    "maximum": 0,
    "exclusiveMaximum": 0,
    "minimum": 0,
    "exclusiveMinimum": 0,
    "maxLength": 0,
    "minLength": 0,
    "pattern": "/regex/",
    "maxItems": 0,
    "minItems": 0,
    "uniqueItems": false,
    "maxContains": 0,
    "minContains": 0,
    "maxProperties": 0,
    "minProperties": 0,
    "required": [],
    "dependentRequired": {
      "property1": [],
      "property2": []
    },
    "const": null,
    "enum": [
      null
    ],
    "type": "array",
    "title": "string",
    "description": "string",
    "default": null,
    "deprecated": false,
    "readOnly": false,
    "writeOnly": false,
    "examples": [
      null
    ],
    "format": "string",
    "contentMediaType": "string",
    "contentEncoding": "string",
    "contentSchema": null
  },
  "history": [
    {
      "id": "string",
      "iterations": [],
      "inputs": {
        "llmApi": "string",
        "llmOutput": "string",
        "messages": [
          {
            "property1": null,
            "property2": null
          }
        ],
        "promptParams": {
          "property1": null,
          "property2": null
        },
        "numReasks": 0,
        "metadata": {
          "property1": null,
          "property2": null
        },
        "fullSchemaReask": true,
        "stream": true,
        "args": [
          true
        ],
        "kwargs": {
          "property1": null,
          "property2": null
        }
      },
      "exception": "string"
    }
  ]
}
```

<h3 id="updateguard-responses">Responses</h3>

|Status|Meaning|Description|Schema|
|---|---|---|---|
|200|[OK](https://tools.ietf.org/html/rfc7231#section-6.3.1)|the updated Guard|[guard](#schemaguard)|
|default|Default|Unexpected error|[HttpError](#schemahttperror)|

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

> 200 Response

```json
{
  "id": "string",
  "name": "string",
  "description": "string",
  "validators": [
    {
      "id": "string",
      "on": "messages",
      "onFail": "exception",
      "args": [
        true
      ],
      "kwargs": {
        "property1": null,
        "property2": null
      }
    }
  ],
  "output_schema": {
    "definitions": {},
    "dependencies": {
      "property1": {},
      "property2": {}
    },
    "$anchor": "string",
    "$ref": "../dictionary",
    "$dynamicRef": "../dictionary",
    "$dynamicAnchor": "string",
    "$vocabulary": {
      "property1": true,
      "property2": true
    },
    "$comment": "string",
    "$defs": {},
    "prefixItems": [
      null
    ],
    "items": null,
    "contains": null,
    "additionalProperties": null,
    "properties": {},
    "patternProperties": {},
    "dependentSchemas": {
      "property1": null,
      "property2": null
    },
    "propertyNames": null,
    "if": null,
    "then": null,
    "else": null,
    "allOf": [
      null
    ],
    "anyOf": [
      null
    ],
    "oneOf": [
      null
    ],
    "not": null,
    "unevaluatedItems": null,
    "unevaluatedProperties": null,
    "multipleOf": 0,
    "maximum": 0,
    "exclusiveMaximum": 0,
    "minimum": 0,
    "exclusiveMinimum": 0,
    "maxLength": 0,
    "minLength": 0,
    "pattern": "/regex/",
    "maxItems": 0,
    "minItems": 0,
    "uniqueItems": false,
    "maxContains": 0,
    "minContains": 0,
    "maxProperties": 0,
    "minProperties": 0,
    "required": [],
    "dependentRequired": {
      "property1": [],
      "property2": []
    },
    "const": null,
    "enum": [
      null
    ],
    "type": "array",
    "title": "string",
    "description": "string",
    "default": null,
    "deprecated": false,
    "readOnly": false,
    "writeOnly": false,
    "examples": [
      null
    ],
    "format": "string",
    "contentMediaType": "string",
    "contentEncoding": "string",
    "contentSchema": null
  },
  "history": [
    {
      "id": "string",
      "iterations": [],
      "inputs": {
        "llmApi": "string",
        "llmOutput": "string",
        "messages": [
          {
            "property1": null,
            "property2": null
          }
        ],
        "promptParams": {
          "property1": null,
          "property2": null
        },
        "numReasks": 0,
        "metadata": {
          "property1": null,
          "property2": null
        },
        "fullSchemaReask": true,
        "stream": true,
        "args": [
          true
        ],
        "kwargs": {
          "property1": null,
          "property2": null
        }
      },
      "exception": "string"
    }
  ]
}
```

<h3 id="deleteguard-responses">Responses</h3>

|Status|Meaning|Description|Schema|
|---|---|---|---|
|200|[OK](https://tools.ietf.org/html/rfc7231#section-6.3.1)|the deleted Guard|[guard](#schemaguard)|
|default|Default|Unexpected error|[HttpError](#schemahttperror)|

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
[
  {
    "id": "string",
    "iterations": [],
    "inputs": {
      "llmApi": "string",
      "llmOutput": "string",
      "messages": [
        {
          "property1": null,
          "property2": null
        }
      ],
      "promptParams": {
        "property1": null,
        "property2": null
      },
      "numReasks": 0,
      "metadata": {
        "property1": null,
        "property2": null
      },
      "fullSchemaReask": true,
      "stream": true,
      "args": [
        true
      ],
      "kwargs": {
        "property1": null,
        "property2": null
      }
    },
    "exception": "string"
  }
]
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
|*anonymous*|[[call](#schemacall)]|false|none|none|
|» Call|[call](#schemacall)|false|none|none|
|»» id|string|true|none|The unique identifier for this Call.  Can be used as an identifier for a specific execution of a Guard.|
|»» iterations|[[iteration](#schemaiteration)]|false|none|none|
|»»» Iteration|[iteration](#schemaiteration)|false|none|none|
|»»»» id|string|true|none|The unique identifier for this Call.  Can be used as an identifier for a specific execution of a Guard.|
|»»»» index|integer|true|none|The zero-based index of this iteration within the current Call.|
|»»»» callId|string|true|none|The unique identifier for the Call that this iteration is a part of.|
|»»»» inputs|[inputs](#schemainputs)|false|none|none|
|»»»»» llmApi|string|false|none|The LLM resource targeted by the user. e.g. openai.chat.completions.create|
|»»»»» llmOutput|string|false|none|The string output from an external LLM call provided by the user via Guard.parse.|
|»»»»» messages|[object]|false|none|The message history for chat models.|
|»»»»»» **additionalProperties**|any|false|none|none|
|»»»»» promptParams|object|false|none|Parameters to be formatted into the prompt.|
|»»»»»» **additionalProperties**|any|false|none|none|
|»»»»» numReasks|integer|false|none|The total number of times the LLM can be called to correct output excluding the initial call.|
|»»»»» metadata|object|false|none|Additional data to be used by Validators during execution time.|
|»»»»»» **additionalProperties**|any|false|none|none|
|»»»»» fullSchemaReask|boolean|false|none|Whether to perform reasks for the entire schema rather than for individual fields.|
|»»»»» stream|boolean|false|none|Whether to use streaming.|
|»»»» outputs|[outputs](#schemaoutputs)|false|none|none|
|»»»»» llmResponseInfo|[LLMResponse](#schemallmresponse)|false|none|Information from the LLM response.|
|»»»»»» promptTokenCount|integer|false|none|none|
|»»»»»» responseTokenCount|integer|false|none|none|
|»»»»»» output|string|true|none|none|
|»»»»»» streamOutput|[string]|false|none|none|
|»»»»»» asyncStreamOutput|[string]|false|none|none|
|»»»»» rawOutput|string|false|none|The string content from the LLM response.|
|»»»»» parsedOutput|any|false|none|none|

*anyOf*

|Name|Type|Required|Restrictions|Description|
|---|---|---|---|---|
|»»»»»» *anonymous*|string|false|none|none|

*or*

|Name|Type|Required|Restrictions|Description|
|---|---|---|---|---|
|»»»»»» *anonymous*|object|false|none|none|
|»»»»»»» **additionalProperties**|any|false|none|none|

*or*

|Name|Type|Required|Restrictions|Description|
|---|---|---|---|---|
|»»»»»» *anonymous*|[anyOf]|false|none|none|
|»»»»»»» AnyType|any|false|none|none|

*anyOf*

|Name|Type|Required|Restrictions|Description|
|---|---|---|---|---|
|»»»»»»»» *anonymous*|any|false|none|none|

*anyOf*

|Name|Type|Required|Restrictions|Description|
|---|---|---|---|---|
|»»»»»»»»» *anonymous*|boolean|false|none|none|

*or*

|Name|Type|Required|Restrictions|Description|
|---|---|---|---|---|
|»»»»»»»»» *anonymous*|integer|false|none|none|

*or*

|Name|Type|Required|Restrictions|Description|
|---|---|---|---|---|
|»»»»»»»»» *anonymous*|null|false|none|none|

*or*

|Name|Type|Required|Restrictions|Description|
|---|---|---|---|---|
|»»»»»»»»» *anonymous*|number|false|none|none|

*or*

|Name|Type|Required|Restrictions|Description|
|---|---|---|---|---|
|»»»»»»»»» *anonymous*|string|false|none|none|

*or*

|Name|Type|Required|Restrictions|Description|
|---|---|---|---|---|
|»»»»»»»» *anonymous*|[objects](#schemaobjects)|false|none|none|
|»»»»»»»»» **additionalProperties**|any|false|none|none|

*or*

|Name|Type|Required|Restrictions|Description|
|---|---|---|---|---|
|»»»»»»»» *anonymous*|[anyOf]|false|none|none|

*anyOf*

|Name|Type|Required|Restrictions|Description|
|---|---|---|---|---|
|»»»»»»»»» *anonymous*|any|false|none|none|

*or*

|Name|Type|Required|Restrictions|Description|
|---|---|---|---|---|
|»»»»»»»»» *anonymous*|[objects](#schemaobjects)|false|none|none|

*continued*

|Name|Type|Required|Restrictions|Description|
|---|---|---|---|---|
|»»»»» validationResponse|any|false|none|none|

*anyOf*

|Name|Type|Required|Restrictions|Description|
|---|---|---|---|---|
|»»»»»» *anonymous*|string|false|none|none|

*or*

|Name|Type|Required|Restrictions|Description|
|---|---|---|---|---|
|»»»»»» *anonymous*|[reask](#schemareask)|false|none|none|
|»»»»»»» **additionalProperties**|any|false|none|none|
|»»»»»»» incorrectValue|any|false|none|none|
|»»»»»»» failResults|[allOf]|false|none|none|
|»»»»»»»» FailResult|[fail-result](#schemafail-result)|false|none|The output from a single Validator.|
|»»»»»»»»» outcome|any|true|none|none|
|»»»»»»»»» errorMessage|string|true|none|none|
|»»»»»»»»» fixValue|any|false|none|none|
|»»»»»»»»» errorSpans|[[error-span](#schemaerror-span)]|false|none|none|
|»»»»»»»»»» ErrorSpan|[error-span](#schemaerror-span)|false|none|none|
|»»»»»»»»»»» start|integer|true|none|none|
|»»»»»»»»»»» end|integer|true|none|none|
|»»»»»»»»»»» reason|string|true|none|The reason validation failed, specific to this chunk.|

*or*

|Name|Type|Required|Restrictions|Description|
|---|---|---|---|---|
|»»»»»» *anonymous*|object|false|none|none|
|»»»»»»» **additionalProperties**|any|false|none|none|

*or*

|Name|Type|Required|Restrictions|Description|
|---|---|---|---|---|
|»»»»»» *anonymous*|[anyOf]|false|none|none|
|»»»»»»» AnyType|any|false|none|none|

*continued*

|Name|Type|Required|Restrictions|Description|
|---|---|---|---|---|
|»»»»» guardedOutput|any|false|none|none|

*anyOf*

|Name|Type|Required|Restrictions|Description|
|---|---|---|---|---|
|»»»»»» *anonymous*|string|false|none|none|

*or*

|Name|Type|Required|Restrictions|Description|
|---|---|---|---|---|
|»»»»»» *anonymous*|object|false|none|none|
|»»»»»»» **additionalProperties**|any|false|none|none|

*or*

|Name|Type|Required|Restrictions|Description|
|---|---|---|---|---|
|»»»»»» *anonymous*|[anyOf]|false|none|none|
|»»»»»»» AnyType|any|false|none|none|

*continued*

|Name|Type|Required|Restrictions|Description|
|---|---|---|---|---|
|»»»»» reasks|[[reask](#schemareask)]|false|none|none|
|»»»»»» ReAsk|[reask](#schemareask)|false|none|none|
|»»»»» validatorLogs|[[validator-log](#schemavalidator-log)]|false|none|none|
|»»»»»» ValidatorLog|[validator-log](#schemavalidator-log)|false|none|none|
|»»»»»»» validatorName|string(PascalCase)|true|none|The class name of the validator.|
|»»»»»»» registeredName|string(kebab-case)|true|none|The registry id of the validator.|
|»»»»»»» instanceId|any|false|none|A unique identifier for the validator that produced this log within the session.|

*anyOf*

|Name|Type|Required|Restrictions|Description|
|---|---|---|---|---|
|»»»»»»»» *anonymous*|string|false|none|none|

*or*

|Name|Type|Required|Restrictions|Description|
|---|---|---|---|---|
|»»»»»»»» *anonymous*|integer|false|none|none|

*continued*

|Name|Type|Required|Restrictions|Description|
|---|---|---|---|---|
|»»»»»»» propertyPath|string|true|none|The JSON path to the property which was validated that produced this log.|
|»»»»»»» valueBeforeValidation|any|true|none|none|
|»»»»»»» valueAfterValidation|any|false|none|none|
|»»»»»»» validationResult|any|false|none|none|

*anyOf*

|Name|Type|Required|Restrictions|Description|
|---|---|---|---|---|
|»»»»»»»» *anonymous*|[pass-result](#schemapass-result)|false|none|The output from a single Validator.|
|»»»»»»»»» outcome|any|true|none|none|
|»»»»»»»»» valueOverride|any|false|none|none|

*or*

|Name|Type|Required|Restrictions|Description|
|---|---|---|---|---|
|»»»»»»»» *anonymous*|[fail-result](#schemafail-result)|false|none|The output from a single Validator.|
|»»»»»»»»» outcome|any|true|none|none|
|»»»»»»»»» errorMessage|string|true|none|none|
|»»»»»»»»» fixValue|any|false|none|none|
|»»»»»»»»» errorSpans|[[error-span](#schemaerror-span)]|false|none|none|

*continued*

|Name|Type|Required|Restrictions|Description|
|---|---|---|---|---|
|»»»»»»» startTime|string(date-time)|false|none|none|
|»»»»»»» endTime|string(date-time)|false|none|none|
|»»»»» error|string|false|none|The error message from any exception which interrupted the Guard execution process.|
|»» inputs|[CallInputs](#schemacallinputs)|false|none|none|

*allOf*

|Name|Type|Required|Restrictions|Description|
|---|---|---|---|---|
|»»» *anonymous*|[inputs](#schemainputs)|false|none|none|

*and*

|Name|Type|Required|Restrictions|Description|
|---|---|---|---|---|
|»»» *anonymous*|[args-and-kwargs](#schemaargs-and-kwargs)|false|none|none|
|»»»» args|[anyOf]|false|none|none|
|»»»»» AnyType|any|false|none|none|
|»»»» kwargs|object|false|none|none|
|»»»»» **additionalProperties**|any|false|none|none|

*continued*

|Name|Type|Required|Restrictions|Description|
|---|---|---|---|---|
|»» exception|string|false|none|none|

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
  "llmApi": "openai.Completion.create",
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
|» llmApi|body|[LLMResource](#schemallmresource)|false|none|

#### Enumerated Values

|Parameter|Value|
|---|---|
|» llmApi|openai.Completion.create|
|» llmApi|openai.completions.create|
|» llmApi|openai.ChatCompletion.create|
|» llmApi|openai.chat.completions.create|
|» llmApi|openai.Completion.acreate|
|» llmApi|openai.ChatCompletion.acreate|
|» llmApi|litellm.completion|
|» llmApi|litellm.acompletion|

> Example responses

> 200 Response

```json
{
  "callId": "string",
  "rawLlmOutput": "string",
  "validatedOutput": "string",
  "reask": {
    "incorrectValue": true,
    "failResults": [
      {
        "outcome": "pass",
        "errorMessage": "string",
        "fixValue": true,
        "errorSpans": [
          {
            "start": 0,
            "end": 0,
            "reason": "string"
          }
        ],
        "metadata": {
          "property1": null,
          "property2": null
        },
        "validatedChunk": true
      }
    ],
    "property1": null,
    "property2": null
  },
  "validationPassed": true,
  "error": "string"
}
```

<h3 id="validate-responses">Responses</h3>

|Status|Meaning|Description|Schema|
|---|---|---|---|
|200|[OK](https://tools.ietf.org/html/rfc7231#section-6.3.1)|The output of the validation|[validation-outcome](#schemavalidation-outcome)|
|default|Default|Unexpected error|[HttpError](#schemahttperror)|

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
|temperature|number|false|none|The sampling temperature|=

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
{
  "callId": "string",
  "rawLlmOutput": "string",
  "validatedOutput": "string",
  "reask": {
    "incorrectValue": true,
    "failResults": [
      {
        "outcome": "pass",
        "errorMessage": "string",
        "fixValue": true,
        "errorSpans": [
          {
            "start": 0,
            "end": 0,
            "reason": "string"
          }
        ],
        "metadata": {
          "property1": null,
          "property2": null
        },
        "validatedChunk": true
      }
    ],
    "property1": null,
    "property2": null
  },
  "validationPassed": true,
  "error": "string"
}

```

#### Properties

*None*

### primitives
<!-- backwards compatibility -->
<a id="schemaprimitives"></a>
<a id="schema_primitives"></a>
<a id="tocSprimitives"></a>
<a id="tocsprimitives"></a>

```json
true

```

#### Properties

anyOf

|Name|Type|Required|Restrictions|Description|
|---|---|---|---|---|
|*anonymous*|boolean|false|none|none|

or

|Name|Type|Required|Restrictions|Description|
|---|---|---|---|---|
|*anonymous*|integer|false|none|none|

or

|Name|Type|Required|Restrictions|Description|
|---|---|---|---|---|
|*anonymous*|null|false|none|none|

or

|Name|Type|Required|Restrictions|Description|
|---|---|---|---|---|
|*anonymous*|number|false|none|none|

or

|Name|Type|Required|Restrictions|Description|
|---|---|---|---|---|
|*anonymous*|string|false|none|none|

### objects
<!-- backwards compatibility -->
<a id="schemaobjects"></a>
<a id="schema_objects"></a>
<a id="tocSobjects"></a>
<a id="tocsobjects"></a>

```json
{
  "property1": null,
  "property2": null
}

```

#### Properties

|Name|Type|Required|Restrictions|Description|
|---|---|---|---|---|
|**additionalProperties**|any|false|none|none|

### arrays
<!-- backwards compatibility -->
<a id="schemaarrays"></a>
<a id="schema_arrays"></a>
<a id="tocSarrays"></a>
<a id="tocsarrays"></a>

```json
[
  true
]

```

#### Properties

anyOf

|Name|Type|Required|Restrictions|Description|
|---|---|---|---|---|
|*anonymous*|[primitives](#schemaprimitives)|false|none|none|

or

|Name|Type|Required|Restrictions|Description|
|---|---|---|---|---|
|*anonymous*|[objects](#schemaobjects)|false|none|none|

### any-type
<!-- backwards compatibility -->
<a id="schemaany-type"></a>
<a id="schema_any-type"></a>
<a id="tocSany-type"></a>
<a id="tocsany-type"></a>

```json
true

```

AnyType

#### Properties

|Name|Type|Required|Restrictions|Description|
|---|---|---|---|---|
|AnyType|any|false|none|none|

anyOf

|Name|Type|Required|Restrictions|Description|
|---|---|---|---|---|
|*anonymous*|[primitives](#schemaprimitives)|false|none|none|

or

|Name|Type|Required|Restrictions|Description|
|---|---|---|---|---|
|*anonymous*|[objects](#schemaobjects)|false|none|none|

or

|Name|Type|Required|Restrictions|Description|
|---|---|---|---|---|
|*anonymous*|[arrays](#schemaarrays)|false|none|none|

### validation-result
<!-- backwards compatibility -->
<a id="schemavalidation-result"></a>
<a id="schema_validation-result"></a>
<a id="tocSvalidation-result"></a>
<a id="tocsvalidation-result"></a>

```json
{
  "outcome": "pass",
  "metadata": {
    "property1": null,
    "property2": null
  },
  "validatedChunk": true
}

```

ValidationResult

#### Properties

|Name|Type|Required|Restrictions|Description|
|---|---|---|---|---|
|outcome|string|true|none|none|
|metadata|object|false|none|none|
|» **additionalProperties**|any|false|none|none|
|validatedChunk|[any-type](#schemaany-type)|false|none|none|

#### Enumerated Values

|Property|Value|
|---|---|
|outcome|pass|
|outcome|fail|

### error-span
<!-- backwards compatibility -->
<a id="schemaerror-span"></a>
<a id="schema_error-span"></a>
<a id="tocSerror-span"></a>
<a id="tocserror-span"></a>

```json
{
  "start": 0,
  "end": 0,
  "reason": "string"
}

```

ErrorSpan

#### Properties

|Name|Type|Required|Restrictions|Description|
|---|---|---|---|---|
|start|integer|true|none|none|
|end|integer|true|none|none|
|reason|string|true|none|The reason validation failed, specific to this chunk.|

### fail-result
<!-- backwards compatibility -->
<a id="schemafail-result"></a>
<a id="schema_fail-result"></a>
<a id="tocSfail-result"></a>
<a id="tocsfail-result"></a>

```json
{
  "outcome": "pass",
  "errorMessage": "string",
  "fixValue": true,
  "errorSpans": [
    {
      "start": 0,
      "end": 0,
      "reason": "string"
    }
  ],
  "metadata": {
    "property1": null,
    "property2": null
  },
  "validatedChunk": true
}

```

FailResult

#### Properties

|Name|Type|Required|Restrictions|Description|
|---|---|---|---|---|
|FailResult|[validation-result](#schemavalidation-result)|false|none|The output from a single Validator.|
|outcome|any|true|none|none|
|errorMessage|string|true|none|none|
|fixValue|[any-type](#schemaany-type)|false|none|none|
|errorSpans|[[error-span](#schemaerror-span)]|false|none|none|

### reask
<!-- backwards compatibility -->
<a id="schemareask"></a>
<a id="schema_reask"></a>
<a id="tocSreask"></a>
<a id="tocsreask"></a>

```json
{
  "incorrectValue": true,
  "failResults": [
    {
      "outcome": "pass",
      "errorMessage": "string",
      "fixValue": true,
      "errorSpans": [
        {
          "start": 0,
          "end": 0,
          "reason": "string"
        }
      ],
      "metadata": {
        "property1": null,
        "property2": null
      },
      "validatedChunk": true
    }
  ],
  "property1": null,
  "property2": null
}

```

ReAsk

#### Properties

|Name|Type|Required|Restrictions|Description|
|---|---|---|---|---|
|**additionalProperties**|any|false|none|none|
|incorrectValue|[any-type](#schemaany-type)|false|none|none|
|failResults|[[fail-result](#schemafail-result)]|false|none|none|

### validation-outcome
<!-- backwards compatibility -->
<a id="schemavalidation-outcome"></a>
<a id="schema_validation-outcome"></a>
<a id="tocSvalidation-outcome"></a>
<a id="tocsvalidation-outcome"></a>

```json
{
  "callId": "string",
  "rawLlmOutput": "string",
  "validatedOutput": "string",
  "reask": {
    "incorrectValue": true,
    "failResults": [
      {
        "outcome": "pass",
        "errorMessage": "string",
        "fixValue": true,
        "errorSpans": [
          {
            "start": 0,
            "end": 0,
            "reason": "string"
          }
        ],
        "metadata": {
          "property1": null,
          "property2": null
        },
        "validatedChunk": true
      }
    ],
    "property1": null,
    "property2": null
  },
  "validationPassed": true,
  "error": "string"
}

```

ValidationOutcome

#### Properties

|Name|Type|Required|Restrictions|Description|
|---|---|---|---|---|
|callId|string|true|none|Foreign key to the most recent Call this resulted from.|
|rawLlmOutput|string|false|none|The raw, unchanged string content from the LLM call.|
|validatedOutput|any|false|none|The validated, and potentially fixed, output from the LLM call after undergoing validation.|

anyOf

|Name|Type|Required|Restrictions|Description|
|---|---|---|---|---|
|» *anonymous*|string|false|none|none|

or

|Name|Type|Required|Restrictions|Description|
|---|---|---|---|---|
|» *anonymous*|object|false|none|none|
|»» **additionalProperties**|any|false|none|none|

or

|Name|Type|Required|Restrictions|Description|
|---|---|---|---|---|
|» *anonymous*|[[any-type](#schemaany-type)]|false|none|none|

continued

|Name|Type|Required|Restrictions|Description|
|---|---|---|---|---|
|reask|[reask](#schemareask)|false|none|If validation continuously fails and all allocated reasks are used, this field will contain the final reask that would have been sent to the LLM if additional reasks were available.|
|validationPassed|boolean|false|none|A boolean to indicate whether or not the LLM output passed validation.  If this is False, the validated_output may be invalid.|
|error|string|false|none|If the validation process raised a handleable exception, this field will contain the error message.|

### Guard
<!-- backwards compatibility -->
<a id="schemaguard"></a>
<a id="schema_Guard"></a>
<a id="tocSguard"></a>
<a id="tocsguard"></a>

```json
{
  "id": "string",
  "name": "string",
  "description": "string",
  "validators": [
    {
      "id": "string",
      "on": "messages",
      "onFail": "exception",
      "args": [
        true
      ],
      "kwargs": {
        "property1": null,
        "property2": null
      }
    }
  ],
  "output_schema": {
    "definitions": {},
    "dependencies": {
      "property1": {},
      "property2": {}
    },
    "$anchor": "string",
    "$ref": "../dictionary",
    "$dynamicRef": "../dictionary",
    "$dynamicAnchor": "string",
    "$vocabulary": {
      "property1": true,
      "property2": true
    },
    "$comment": "string",
    "$defs": {},
    "prefixItems": [
      null
    ],
    "items": null,
    "contains": null,
    "additionalProperties": null,
    "properties": {},
    "patternProperties": {},
    "dependentSchemas": {
      "property1": null,
      "property2": null
    },
    "propertyNames": null,
    "if": null,
    "then": null,
    "else": null,
    "allOf": [
      null
    ],
    "anyOf": [
      null
    ],
    "oneOf": [
      null
    ],
    "not": null,
    "unevaluatedItems": null,
    "unevaluatedProperties": null,
    "multipleOf": 0,
    "maximum": 0,
    "exclusiveMaximum": 0,
    "minimum": 0,
    "exclusiveMinimum": 0,
    "maxLength": 0,
    "minLength": 0,
    "pattern": "/regex/",
    "maxItems": 0,
    "minItems": 0,
    "uniqueItems": false,
    "maxContains": 0,
    "minContains": 0,
    "maxProperties": 0,
    "minProperties": 0,
    "required": [],
    "dependentRequired": {
      "property1": [],
      "property2": []
    },
    "const": null,
    "enum": [
      null
    ],
    "type": "array",
    "title": "string",
    "description": "string",
    "default": null,
    "deprecated": false,
    "readOnly": false,
    "writeOnly": false,
    "examples": [
      null
    ],
    "format": "string",
    "contentMediaType": "string",
    "contentEncoding": "string",
    "contentSchema": null
  },
  "history": [
    {
      "id": "string",
      "iterations": [],
      "inputs": {
        "llmApi": "string",
        "llmOutput": "string",
        "messages": [
          {
            "property1": null,
            "property2": null
          }
        ],
        "promptParams": {
          "property1": null,
          "property2": null
        },
        "numReasks": 0,
        "metadata": {
          "property1": null,
          "property2": null
        },
        "fullSchemaReask": true,
        "stream": true,
        "args": [
          true
        ],
        "kwargs": {
          "property1": null,
          "property2": null
        }
      },
      "exception": "string"
    }
  ]
}

```

#### Properties

*None*

### schemaArray
<!-- backwards compatibility -->
<a id="schemaschemaarray"></a>
<a id="schema_schemaArray"></a>
<a id="tocSschemaarray"></a>
<a id="tocsschemaarray"></a>

```json
[
  null
]

```

#### Properties

*None*

### nonNegativeInteger
<!-- backwards compatibility -->
<a id="schemanonnegativeinteger"></a>
<a id="schema_nonNegativeInteger"></a>
<a id="tocSnonnegativeinteger"></a>
<a id="tocsnonnegativeinteger"></a>

```json
0

```

#### Properties

|Name|Type|Required|Restrictions|Description|
|---|---|---|---|---|
|*anonymous*|integer|false|none|none|

### nonNegativeIntegerDefault0
<!-- backwards compatibility -->
<a id="schemanonnegativeintegerdefault0"></a>
<a id="schema_nonNegativeIntegerDefault0"></a>
<a id="tocSnonnegativeintegerdefault0"></a>
<a id="tocsnonnegativeintegerdefault0"></a>

```json
0

```

#### Properties

*None*

### simpleTypes
<!-- backwards compatibility -->
<a id="schemasimpletypes"></a>
<a id="schema_simpleTypes"></a>
<a id="tocSsimpletypes"></a>
<a id="tocssimpletypes"></a>

```json
"array"

```

#### Properties

*None*

### stringArray
<!-- backwards compatibility -->
<a id="schemastringarray"></a>
<a id="schema_stringArray"></a>
<a id="tocSstringarray"></a>
<a id="tocsstringarray"></a>

```json
[]

```

#### Properties

*None*

### LLMResponse
<!-- backwards compatibility -->
<a id="schemallmresponse"></a>
<a id="schema_LLMResponse"></a>
<a id="tocSllmresponse"></a>
<a id="tocsllmresponse"></a>

```json
{
  "promptTokenCount": 0,
  "responseTokenCount": 0,
  "output": "string",
  "streamOutput": [
    "string"
  ],
  "asyncStreamOutput": [
    "string"
  ]
}

```

Information from the LLM response.

#### Properties

|Name|Type|Required|Restrictions|Description|
|---|---|---|---|---|
|promptTokenCount|integer|false|none|none|
|responseTokenCount|integer|false|none|none|
|output|string|true|none|none|
|streamOutput|[string]|false|none|none|
|asyncStreamOutput|[string]|false|none|none|

### CallInputs
<!-- backwards compatibility -->
<a id="schemacallinputs"></a>
<a id="schema_CallInputs"></a>
<a id="tocScallinputs"></a>
<a id="tocscallinputs"></a>

```json
{
  "llmApi": "string",
  "llmOutput": "string",
  "messages": [
    {
      "property1": null,
      "property2": null
    }
  ],
  "promptParams": {
    "property1": null,
    "property2": null
  },
  "numReasks": 0,
  "metadata": {
    "property1": null,
    "property2": null
  },
  "fullSchemaReask": true,
  "stream": true,
  "args": [
    true
  ],
  "kwargs": {
    "property1": null,
    "property2": null
  }
}

```

CallInputs

#### Properties

allOf

|Name|Type|Required|Restrictions|Description|
|---|---|---|---|---|
|*anonymous*|[inputs](#schemainputs)|false|none|none|

and

|Name|Type|Required|Restrictions|Description|
|---|---|---|---|---|
|*anonymous*|[args-and-kwargs](#schemaargs-and-kwargs)|false|none|none|

### ValidatorReference
<!-- backwards compatibility -->
<a id="schemavalidatorreference"></a>
<a id="schema_ValidatorReference"></a>
<a id="tocSvalidatorreference"></a>
<a id="tocsvalidatorreference"></a>

```json
{
  "id": "string",
  "on": "messages",
  "onFail": "exception",
  "args": [
    true
  ],
  "kwargs": {
    "property1": null,
    "property2": null
  }
}

```

ValidatorReference

#### Properties

|Name|Type|Required|Restrictions|Description|
|---|---|---|---|---|
|ValidatorReference|[args-and-kwargs](#schemaargs-and-kwargs)|false|none|none|
|id|string|true|none|The unique identifier for this Validator.  Often the hub id; e.g. guardrails/regex_match|
|on|any|false|none|A reference to the property this validator should be applied against.  Can be a valid JSON path or a meta-property such as "prompt" or "output"|

anyOf

|Name|Type|Required|Restrictions|Description|
|---|---|---|---|---|
|» *anonymous*|string|false|none|meta-property|

or

|Name|Type|Required|Restrictions|Description|
|---|---|---|---|---|
|» *anonymous*|string|false|none|JSON path to property; e.g. $.foo.bar|

continued

|Name|Type|Required|Restrictions|Description|
|---|---|---|---|---|
|onFail|string|false|none|none|

#### Enumerated Values

|Property|Value|
|---|---|
|*anonymous*|prompt|
|*anonymous*|messages|
|*anonymous*|output|
|onFail|exception|
|onFail|filter|
|onFail|fix|
|onFail|fix_reask|
|onFail|noop|
|onFail|reask|
|onFail|refrain|
|onFail|custom|

### args-and-kwargs
<!-- backwards compatibility -->
<a id="schemaargs-and-kwargs"></a>
<a id="schema_args-and-kwargs"></a>
<a id="tocSargs-and-kwargs"></a>
<a id="tocsargs-and-kwargs"></a>

```json
{
  "args": [
    true
  ],
  "kwargs": {
    "property1": null,
    "property2": null
  }
}

```

ArgsAndKwargs

#### Properties

|Name|Type|Required|Restrictions|Description|
|---|---|---|---|---|
|args|[[any-type](#schemaany-type)]|false|none|none|
|kwargs|object|false|none|none|
|» **additionalProperties**|any|false|none|none|

### core
<!-- backwards compatibility -->
<a id="schemacore"></a>
<a id="schema_core"></a>
<a id="tocScore"></a>
<a id="tocscore"></a>

```json
{
  "$anchor": "string",
  "$ref": "../dictionary",
  "$dynamicRef": "../dictionary",
  "$dynamicAnchor": "string",
  "$vocabulary": {
    "property1": true,
    "property2": true
  },
  "$comment": "string",
  "$defs": {}
}

```

Core vocabulary meta-schema

#### Properties

|Name|Type|Required|Restrictions|Description|
|---|---|---|---|---|
|Core vocabulary meta-schema|object,boolean|false|none|none|
|$anchor|string|false|none|none|
|$ref|string(uri-reference)|false|none|none|
|$dynamicRef|string(uri-reference)|false|none|none|
|$dynamicAnchor|string|false|none|none|
|$vocabulary|object|false|none|none|
|» **additionalProperties**|boolean|false|none|none|
|$comment|string|false|none|none|
|$defs|object|false|none|none|
|» **additionalProperties**|any|false|none|none|

### applicator
<!-- backwards compatibility -->
<a id="schemaapplicator"></a>
<a id="schema_applicator"></a>
<a id="tocSapplicator"></a>
<a id="tocsapplicator"></a>

```json
{
  "prefixItems": [
    null
  ],
  "items": null,
  "contains": null,
  "additionalProperties": null,
  "properties": {},
  "patternProperties": {},
  "dependentSchemas": {
    "property1": null,
    "property2": null
  },
  "propertyNames": null,
  "if": null,
  "then": null,
  "else": null,
  "allOf": [
    null
  ],
  "anyOf": [
    null
  ],
  "oneOf": [
    null
  ],
  "not": null
}

```

Applicator vocabulary meta-schema

#### Properties

|Name|Type|Required|Restrictions|Description|
|---|---|---|---|---|
|Applicator vocabulary meta-schema|object,boolean|false|none|none|
|prefixItems|[schemaArray](#schemaschemaarray)|false|none|none|
|items|any|false|none|none|
|contains|any|false|none|none|
|additionalProperties|any|false|none|none|
|properties|object|false|none|none|
|» **additionalProperties**|any|false|none|none|
|patternProperties|object|false|none|none|
|» **additionalProperties**|any|false|none|none|
|dependentSchemas|object|false|none|none|
|» **additionalProperties**|any|false|none|none|
|propertyNames|any|false|none|none|
|if|any|false|none|none|
|then|any|false|none|none|
|else|any|false|none|none|
|allOf|[schemaArray](#schemaschemaarray)|false|none|none|
|anyOf|[schemaArray](#schemaschemaarray)|false|none|none|
|oneOf|[schemaArray](#schemaschemaarray)|false|none|none|
|not|any|false|none|none|

### unevaluated
<!-- backwards compatibility -->
<a id="schemaunevaluated"></a>
<a id="schema_unevaluated"></a>
<a id="tocSunevaluated"></a>
<a id="tocsunevaluated"></a>

```json
{
  "unevaluatedItems": null,
  "unevaluatedProperties": null
}

```

Unevaluated applicator vocabulary meta-schema

#### Properties

|Name|Type|Required|Restrictions|Description|
|---|---|---|---|---|
|Unevaluated applicator vocabulary meta-schema|object,boolean|false|none|none|
|unevaluatedItems|any|false|none|none|
|unevaluatedProperties|any|false|none|none|

### validation
<!-- backwards compatibility -->
<a id="schemavalidation"></a>
<a id="schema_validation"></a>
<a id="tocSvalidation"></a>
<a id="tocsvalidation"></a>

```json
{
  "multipleOf": 0,
  "maximum": 0,
  "exclusiveMaximum": 0,
  "minimum": 0,
  "exclusiveMinimum": 0,
  "maxLength": 0,
  "minLength": 0,
  "pattern": "/regex/",
  "maxItems": 0,
  "minItems": 0,
  "uniqueItems": false,
  "maxContains": 0,
  "minContains": 0,
  "maxProperties": 0,
  "minProperties": 0,
  "required": [],
  "dependentRequired": {
    "property1": [],
    "property2": []
  },
  "const": null,
  "enum": [
    null
  ],
  "type": "array"
}

```

Validation vocabulary meta-schema

#### Properties

|Name|Type|Required|Restrictions|Description|
|---|---|---|---|---|
|Validation vocabulary meta-schema|object,boolean|false|none|none|
|multipleOf|number|false|none|none|
|maximum|number|false|none|none|
|exclusiveMaximum|number|false|none|none|
|minimum|number|false|none|none|
|exclusiveMinimum|number|false|none|none|
|maxLength|[nonNegativeInteger](#schemanonnegativeinteger)|false|none|none|
|minLength|[nonNegativeIntegerDefault0](#schemanonnegativeintegerdefault0)|false|none|none|
|pattern|string(regex)|false|none|none|
|maxItems|[nonNegativeInteger](#schemanonnegativeinteger)|false|none|none|
|minItems|[nonNegativeIntegerDefault0](#schemanonnegativeintegerdefault0)|false|none|none|
|uniqueItems|boolean|false|none|none|
|maxContains|[nonNegativeInteger](#schemanonnegativeinteger)|false|none|none|
|minContains|[nonNegativeInteger](#schemanonnegativeinteger)|false|none|none|
|maxProperties|[nonNegativeInteger](#schemanonnegativeinteger)|false|none|none|
|minProperties|[nonNegativeIntegerDefault0](#schemanonnegativeintegerdefault0)|false|none|none|
|required|[stringArray](#schemastringarray)|false|none|none|
|dependentRequired|object|false|none|none|
|» **additionalProperties**|[stringArray](#schemastringarray)|false|none|none|
|const|any|false|none|none|
|enum|[any]|false|none|none|
|type|any|false|none|none|

anyOf

|Name|Type|Required|Restrictions|Description|
|---|---|---|---|---|
|» *anonymous*|[simpleTypes](#schemasimpletypes)|false|none|none|

or

|Name|Type|Required|Restrictions|Description|
|---|---|---|---|---|
|» *anonymous*|[[simpleTypes](#schemasimpletypes)]|false|none|none|

#### meta-data
<!-- backwards compatibility -->
<a id="schemameta-data"></a>
<a id="schema_meta-data"></a>
<a id="tocSmeta-data"></a>
<a id="tocsmeta-data"></a>

```json
{
  "title": "string",
  "description": "string",
  "default": null,
  "deprecated": false,
  "readOnly": false,
  "writeOnly": false,
  "examples": [
    null
  ]
}

```

Meta-data vocabulary meta-schema

#### Properties

|Name|Type|Required|Restrictions|Description|
|---|---|---|---|---|
|Meta-data vocabulary meta-schema|object,boolean|false|none|none|
|title|string|false|none|none|
|description|string|false|none|none|
|default|any|false|none|none|
|deprecated|boolean|false|none|none|
|readOnly|boolean|false|none|none|
|writeOnly|boolean|false|none|none|
|examples|[any]|false|none|none|

### format-annotation
<!-- backwards compatibility -->
<a id="schemaformat-annotation"></a>
<a id="schema_format-annotation"></a>
<a id="tocSformat-annotation"></a>
<a id="tocsformat-annotation"></a>

```json
{
  "format": "string"
}

```

Format vocabulary meta-schema for annotation results

#### Properties

|Name|Type|Required|Restrictions|Description|
|---|---|---|---|---|
|Format vocabulary meta-schema for annotation results|object,boolean|false|none|none|
|format|string|false|none|none|

### content
<!-- backwards compatibility -->
<a id="schemacontent"></a>
<a id="schema_content"></a>
<a id="tocScontent"></a>
<a id="tocscontent"></a>

```json
{
  "contentMediaType": "string",
  "contentEncoding": "string",
  "contentSchema": null
}

```

Content vocabulary meta-schema

#### Properties

|Name|Type|Required|Restrictions|Description|
|---|---|---|---|---|
|Content vocabulary meta-schema|object,boolean|false|none|none|
|contentMediaType|string|false|none|none|
|contentEncoding|string|false|none|none|
|contentSchema|any|false|none|none|

### schema
<!-- backwards compatibility -->
<a id="schemaschema"></a>
<a id="schema_schema"></a>
<a id="tocSschema"></a>
<a id="tocsschema"></a>

```json
{
  "definitions": {},
  "dependencies": {
    "property1": {},
    "property2": {}
  },
  "$anchor": "string",
  "$ref": "../dictionary",
  "$dynamicRef": "../dictionary",
  "$dynamicAnchor": "string",
  "$vocabulary": {
    "property1": true,
    "property2": true
  },
  "$comment": "string",
  "$defs": {},
  "prefixItems": [
    null
  ],
  "items": null,
  "contains": null,
  "additionalProperties": null,
  "properties": {},
  "patternProperties": {},
  "dependentSchemas": {
    "property1": null,
    "property2": null
  },
  "propertyNames": null,
  "if": null,
  "then": null,
  "else": null,
  "allOf": [
    null
  ],
  "anyOf": [
    null
  ],
  "oneOf": [
    null
  ],
  "not": null,
  "unevaluatedItems": null,
  "unevaluatedProperties": null,
  "multipleOf": 0,
  "maximum": 0,
  "exclusiveMaximum": 0,
  "minimum": 0,
  "exclusiveMinimum": 0,
  "maxLength": 0,
  "minLength": 0,
  "pattern": "/regex/",
  "maxItems": 0,
  "minItems": 0,
  "uniqueItems": false,
  "maxContains": 0,
  "minContains": 0,
  "maxProperties": 0,
  "minProperties": 0,
  "required": [],
  "dependentRequired": {
    "property1": [],
    "property2": []
  },
  "const": null,
  "enum": [
    null
  ],
  "type": "array",
  "title": "string",
  "description": "string",
  "default": null,
  "deprecated": false,
  "readOnly": false,
  "writeOnly": false,
  "examples": [
    null
  ],
  "format": "string",
  "contentMediaType": "string",
  "contentEncoding": "string",
  "contentSchema": null
}

```

Core and Validation specifications meta-schema

#### Properties

|Name|Type|Required|Restrictions|Description|
|---|---|---|---|---|
|Core and Validation specifications meta-schema|object,boolean|false|none|none|
|definitions|object|false|none|none|
|» **additionalProperties**|any|false|none|none|
|dependencies|object|false|none|none|
|» **additionalProperties**|any|false|none|none|

anyOf

|Name|Type|Required|Restrictions|Description|
|---|---|---|---|---|
|»» *anonymous*|any|false|none|none|

or

|Name|Type|Required|Restrictions|Description|
|---|---|---|---|---|
|»» *anonymous*|[validation/definitions/stringArray](#schemavalidation/definitions/stringarray)|false|none|none|

allOf

|Name|Type|Required|Restrictions|Description|
|---|---|---|---|---|
|*anonymous*|[core](#schemacore)|false|none|none|

and

|Name|Type|Required|Restrictions|Description|
|---|---|---|---|---|
|*anonymous*|[applicator](#schemaapplicator)|false|none|none|

and

|Name|Type|Required|Restrictions|Description|
|---|---|---|---|---|
|*anonymous*|[unevaluated](#schemaunevaluated)|false|none|none|

and

|Name|Type|Required|Restrictions|Description|
|---|---|---|---|---|
|*anonymous*|[validation](#schemavalidation)|false|none|none|

and

|Name|Type|Required|Restrictions|Description|
|---|---|---|---|---|
|*anonymous*|[meta-data](#schemameta-data)|false|none|none|

and

|Name|Type|Required|Restrictions|Description|
|---|---|---|---|---|
|*anonymous*|[format-annotation](#schemaformat-annotation)|false|none|none|

and

|Name|Type|Required|Restrictions|Description|
|---|---|---|---|---|
|*anonymous*|[content](#schemacontent)|false|none|none|

### inputs
<!-- backwards compatibility -->
<a id="schemainputs"></a>
<a id="schema_inputs"></a>
<a id="tocSinputs"></a>
<a id="tocsinputs"></a>

```json
{
  "llmApi": "string",
  "llmOutput": "string",
  "messages": [
    {
      "property1": null,
      "property2": null
    }
  ],
  "promptParams": {
    "property1": null,
    "property2": null
  },
  "numReasks": 0,
  "metadata": {
    "property1": null,
    "property2": null
  },
  "fullSchemaReask": true,
  "stream": true
}

```

Inputs

#### Properties

|Name|Type|Required|Restrictions|Description|
|---|---|---|---|---|
|llmApi|string|false|none|The LLM resource targeted by the user. e.g. openai.chat.completions.create|
|llmOutput|string|false|none|The string output from an external LLM call provided by the user via Guard.parse.|
|messages|[object]|false|none|The message history for chat models.|
|» **additionalProperties**|any|false|none|none|
|promptParams|object|false|none|Parameters to be formatted into the prompt.|
|» **additionalProperties**|any|false|none|none|
|numReasks|integer|false|none|The total number of times the LLM can be called to correct output excluding the initial call.|
|metadata|object|false|none|Additional data to be used by Validators during execution time.|
|» **additionalProperties**|any|false|none|none|
|fullSchemaReask|boolean|false|none|Whether to perform reasks for the entire schema rather than for individual fields.|
|stream|boolean|false|none|Whether to use streaming.|

### pass-result
<!-- backwards compatibility -->
<a id="schemapass-result"></a>
<a id="schema_pass-result"></a>
<a id="tocSpass-result"></a>
<a id="tocspass-result"></a>

```json
{
  "outcome": "pass",
  "valueOverride": true,
  "metadata": {
    "property1": null,
    "property2": null
  },
  "validatedChunk": true
}

```

PassResult

#### Properties

|Name|Type|Required|Restrictions|Description|
|---|---|---|---|---|
|PassResult|[validation-result](#schemavalidation-result)|false|none|The output from a single Validator.|
|outcome|any|true|none|none|
|valueOverride|[any-type](#schemaany-type)|false|none|none|

#### validator-log
<!-- backwards compatibility -->
<a id="schemavalidator-log"></a>
<a id="schema_validator-log"></a>
<a id="tocSvalidator-log"></a>
<a id="tocsvalidator-log"></a>

```json
{
  "validatorName": "string",
  "registeredName": "string",
  "instanceId": "string",
  "propertyPath": "string",
  "valueBeforeValidation": true,
  "valueAfterValidation": true,
  "validationResult": {
    "outcome": "pass",
    "valueOverride": true,
    "metadata": {
      "property1": null,
      "property2": null
    },
    "validatedChunk": true
  },
  "startTime": "2019-08-24T14:15:22Z",
  "endTime": "2019-08-24T14:15:22Z"
}

```

ValidatorLog

#### Properties

|Name|Type|Required|Restrictions|Description|
|---|---|---|---|---|
|validatorName|string(PascalCase)|true|none|The class name of the validator.|
|registeredName|string(kebab-case)|true|none|The registry id of the validator.|
|instanceId|any|false|none|A unique identifier for the validator that produced this log within the session.|

anyOf

|Name|Type|Required|Restrictions|Description|
|---|---|---|---|---|
|» *anonymous*|string|false|none|none|

or

|Name|Type|Required|Restrictions|Description|
|---|---|---|---|---|
|» *anonymous*|integer|false|none|none|

continued

|Name|Type|Required|Restrictions|Description|
|---|---|---|---|---|
|propertyPath|string|true|none|The JSON path to the property which was validated that produced this log.|
|valueBeforeValidation|[any-type](#schemaany-type)|true|none|none|
|valueAfterValidation|[any-type](#schemaany-type)|false|none|none|
|validationResult|any|false|none|none|

anyOf

|Name|Type|Required|Restrictions|Description|
|---|---|---|---|---|
|» *anonymous*|[pass-result](#schemapass-result)|false|none|none|

or

|Name|Type|Required|Restrictions|Description|
|---|---|---|---|---|
|» *anonymous*|[fail-result](#schemafail-result)|false|none|none|

continued

|Name|Type|Required|Restrictions|Description|
|---|---|---|---|---|
|startTime|string(date-time)|false|none|none|
|endTime|string(date-time)|false|none|none|

### outputs
<!-- backwards compatibility -->
<a id="schemaoutputs"></a>
<a id="schema_outputs"></a>
<a id="tocSoutputs"></a>
<a id="tocsoutputs"></a>

```json
{
  "llmResponseInfo": {
    "promptTokenCount": 0,
    "responseTokenCount": 0,
    "output": "string",
    "streamOutput": [
      "string"
    ],
    "asyncStreamOutput": [
      "string"
    ]
  },
  "rawOutput": "string",
  "parsedOutput": "string",
  "validationResponse": "string",
  "guardedOutput": "string",
  "reasks": [
    {
      "incorrectValue": true,
      "failResults": [
        {
          "outcome": "pass",
          "errorMessage": "string",
          "fixValue": true,
          "errorSpans": [
            {
              "start": 0,
              "end": 0,
              "reason": "string"
            }
          ],
          "metadata": {
            "property1": null,
            "property2": null
          },
          "validatedChunk": true
        }
      ],
      "property1": null,
      "property2": null
    }
  ],
  "validatorLogs": [
    {
      "validatorName": "string",
      "registeredName": "string",
      "instanceId": "string",
      "propertyPath": "string",
      "valueBeforeValidation": true,
      "valueAfterValidation": true,
      "validationResult": {
        "outcome": "pass",
        "valueOverride": true,
        "metadata": {
          "property1": null,
          "property2": null
        },
        "validatedChunk": true
      },
      "startTime": "2019-08-24T14:15:22Z",
      "endTime": "2019-08-24T14:15:22Z"
    }
  ],
  "error": "string"
}

```

Outputs

#### Properties

|Name|Type|Required|Restrictions|Description|
|---|---|---|---|---|
|llmResponseInfo|[LLMResponse](#schemallmresponse)|false|none|Information from the LLM response.|
|rawOutput|string|false|none|The string content from the LLM response.|
|parsedOutput|any|false|none|none|

anyOf

|Name|Type|Required|Restrictions|Description|
|---|---|---|---|---|
|» *anonymous*|string|false|none|none|

or

|Name|Type|Required|Restrictions|Description|
|---|---|---|---|---|
|» *anonymous*|object|false|none|none|
|»» **additionalProperties**|any|false|none|none|

or

|Name|Type|Required|Restrictions|Description|
|---|---|---|---|---|
|» *anonymous*|[[any-type](#schemaany-type)]|false|none|none|

continued

|Name|Type|Required|Restrictions|Description|
|---|---|---|---|---|
|validationResponse|any|false|none|none|

anyOf

|Name|Type|Required|Restrictions|Description|
|---|---|---|---|---|
|» *anonymous*|string|false|none|none|

or

|Name|Type|Required|Restrictions|Description|
|---|---|---|---|---|
|» *anonymous*|[reask](#schemareask)|false|none|none|

or

|Name|Type|Required|Restrictions|Description|
|---|---|---|---|---|
|» *anonymous*|object|false|none|none|
|»» **additionalProperties**|any|false|none|none|

or

|Name|Type|Required|Restrictions|Description|
|---|---|---|---|---|
|» *anonymous*|[[any-type](#schemaany-type)]|false|none|none|

continued

|Name|Type|Required|Restrictions|Description|
|---|---|---|---|---|
|guardedOutput|any|false|none|none|

anyOf

|Name|Type|Required|Restrictions|Description|
|---|---|---|---|---|
|» *anonymous*|string|false|none|none|

or

|Name|Type|Required|Restrictions|Description|
|---|---|---|---|---|
|» *anonymous*|object|false|none|none|
|»» **additionalProperties**|any|false|none|none|

or

|Name|Type|Required|Restrictions|Description|
|---|---|---|---|---|
|» *anonymous*|[[any-type](#schemaany-type)]|false|none|none|

continued

|Name|Type|Required|Restrictions|Description|
|---|---|---|---|---|
|reasks|[[reask](#schemareask)]|false|none|none|
|validatorLogs|[[validator-log](#schemavalidator-log)]|false|none|none|
|error|string|false|none|The error message from any exception which interrupted the Guard execution process.|

### iteration
<!-- backwards compatibility -->
<a id="schemaiteration"></a>
<a id="schema_iteration"></a>
<a id="tocSiteration"></a>
<a id="tocsiteration"></a>

```json
{
  "id": "string",
  "index": 0,
  "callId": "string",
  "inputs": {
    "llmApi": "string",
    "llmOutput": "string",
    "messages": [
      {
        "property1": null,
        "property2": null
      }
    ],
    "promptParams": {
      "property1": null,
      "property2": null
    },
    "numReasks": 0,
    "metadata": {
      "property1": null,
      "property2": null
    },
    "fullSchemaReask": true,
    "stream": true
  },
  "outputs": {
    "llmResponseInfo": {
      "promptTokenCount": 0,
      "responseTokenCount": 0,
      "output": "string",
      "streamOutput": [
        "string"
      ],
      "asyncStreamOutput": [
        "string"
      ]
    },
    "rawOutput": "string",
    "parsedOutput": "string",
    "validationResponse": "string",
    "guardedOutput": "string",
    "reasks": [
      {
        "incorrectValue": true,
        "failResults": [
          {
            "outcome": "pass",
            "errorMessage": "string",
            "fixValue": true,
            "errorSpans": [
              {
                "start": 0,
                "end": 0,
                "reason": "string"
              }
            ],
            "metadata": {
              "property1": null,
              "property2": null
            },
            "validatedChunk": true
          }
        ],
        "property1": null,
        "property2": null
      }
    ],
    "validatorLogs": [
      {
        "validatorName": "string",
        "registeredName": "string",
        "instanceId": "string",
        "propertyPath": "string",
        "valueBeforeValidation": true,
        "valueAfterValidation": true,
        "validationResult": {
          "outcome": "pass",
          "valueOverride": true,
          "metadata": {
            "property1": null,
            "property2": null
          },
          "validatedChunk": true
        },
        "startTime": "2019-08-24T14:15:22Z",
        "endTime": "2019-08-24T14:15:22Z"
      }
    ],
    "error": "string"
  }
}

```

Iteration

#### Properties

|Name|Type|Required|Restrictions|Description|
|---|---|---|---|---|
|id|string|true|none|The unique identifier for this Call.  Can be used as an identifier for a specific execution of a Guard.|
|index|integer|true|none|The zero-based index of this iteration within the current Call.|
|callId|string|true|none|The unique identifier for the Call that this iteration is a part of.|
|inputs|[inputs](#schemainputs)|false|none|none|
|outputs|[outputs](#schemaoutputs)|false|none|none|

### call
<!-- backwards compatibility -->
<a id="schemacall"></a>
<a id="schema_call"></a>
<a id="tocScall"></a>
<a id="tocscall"></a>

```json
{
  "id": "string",
  "iterations": [],
  "inputs": {
    "llmApi": "string",
    "llmOutput": "string",
    "messages": [
      {
        "property1": null,
        "property2": null
      }
    ],
    "promptParams": {
      "property1": null,
      "property2": null
    },
    "numReasks": 0,
    "metadata": {
      "property1": null,
      "property2": null
    },
    "fullSchemaReask": true,
    "stream": true,
    "args": [
      true
    ],
    "kwargs": {
      "property1": null,
      "property2": null
    }
  },
  "exception": "string"
}

```

Call

#### Properties

|Name|Type|Required|Restrictions|Description|
|---|---|---|---|---|
|id|string|true|none|The unique identifier for this Call.  Can be used as an identifier for a specific execution of a Guard.|
|iterations|[[iteration](#schemaiteration)]|false|none|none|
|inputs|[CallInputs](#schemacallinputs)|false|none|none|
|exception|string|false|none|none|

### guard
<!-- backwards compatibility -->
<a id="schemaguard"></a>
<a id="schema_guard"></a>
<a id="tocSguard"></a>
<a id="tocsguard"></a>

```json
{
  "id": "string",
  "name": "string",
  "description": "string",
  "validators": [
    {
      "id": "string",
      "on": "messages",
      "onFail": "exception",
      "args": [
        true
      ],
      "kwargs": {
        "property1": null,
        "property2": null
      }
    }
  ],
  "output_schema": {
    "definitions": {},
    "dependencies": {
      "property1": {},
      "property2": {}
    },
    "$anchor": "string",
    "$ref": "../dictionary",
    "$dynamicRef": "../dictionary",
    "$dynamicAnchor": "string",
    "$vocabulary": {
      "property1": true,
      "property2": true
    },
    "$comment": "string",
    "$defs": {},
    "prefixItems": [
      null
    ],
    "items": null,
    "contains": null,
    "additionalProperties": null,
    "properties": {},
    "patternProperties": {},
    "dependentSchemas": {
      "property1": null,
      "property2": null
    },
    "propertyNames": null,
    "if": null,
    "then": null,
    "else": null,
    "allOf": [
      null
    ],
    "anyOf": [
      null
    ],
    "oneOf": [
      null
    ],
    "not": null,
    "unevaluatedItems": null,
    "unevaluatedProperties": null,
    "multipleOf": 0,
    "maximum": 0,
    "exclusiveMaximum": 0,
    "minimum": 0,
    "exclusiveMinimum": 0,
    "maxLength": 0,
    "minLength": 0,
    "pattern": "/regex/",
    "maxItems": 0,
    "minItems": 0,
    "uniqueItems": false,
    "maxContains": 0,
    "minContains": 0,
    "maxProperties": 0,
    "minProperties": 0,
    "required": [],
    "dependentRequired": {
      "property1": [],
      "property2": []
    },
    "const": null,
    "enum": [
      null
    ],
    "type": "array",
    "title": "string",
    "description": "string",
    "default": null,
    "deprecated": false,
    "readOnly": false,
    "writeOnly": false,
    "examples": [
      null
    ],
    "format": "string",
    "contentMediaType": "string",
    "contentEncoding": "string",
    "contentSchema": null
  },
  "history": [
    {
      "id": "string",
      "iterations": [],
      "inputs": {
        "llmApi": "string",
        "llmOutput": "string",
        "messages": [
          {
            "property1": null,
            "property2": null
          }
        ],
        "promptParams": {
          "property1": null,
          "property2": null
        },
        "numReasks": 0,
        "metadata": {
          "property1": null,
          "property2": null
        },
        "fullSchemaReask": true,
        "stream": true,
        "args": [
          true
        ],
        "kwargs": {
          "property1": null,
          "property2": null
        }
      },
      "exception": "string"
    }
  ]
}

```

Guard

#### Properties

|Name|Type|Required|Restrictions|Description|
|---|---|---|---|---|
|id|string(kebab-case)|true|none|The unique identifier for the Guard.|
|name|string|true|none|The name for the Guard.|
|description|string|false|none|A description that concisely states the expected behaviour or purpose of the Guard.|
|validators|[[ValidatorReference](#schemavalidatorreference)]|false|none|none|
|output_schema|[schema](#schemaschema)|false|none|none|
|history|[[call](#schemacall)]|false|read-only|none|

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
  "llmApi": "openai.Completion.create",
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
|llmApi|[LLMResource](#schemallmresource)|false|none|none|

### LLMResource
<!-- backwards compatibility -->
<a id="schemallmresource"></a>
<a id="schema_LLMResource"></a>
<a id="tocSllmresource"></a>
<a id="tocsllmresource"></a>

```json
"openai.Completion.create"

```

#### Properties

|Name|Type|Required|Restrictions|Description|
|---|---|---|---|---|
|*anonymous*|string|false|none|none|

#### Enumerated Values

|Property|Value|
|---|---|
|*anonymous*|openai.Completion.create|
|*anonymous*|openai.completions.create|
|*anonymous*|openai.ChatCompletion.create|
|*anonymous*|openai.chat.completions.create|
|*anonymous*|openai.Completion.acreate|
|*anonymous*|openai.ChatCompletion.acreate|
|*anonymous*|litellm.completion|
|*anonymous*|litellm.acompletion|
