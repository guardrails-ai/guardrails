# Guardrails Features and Functionalities

Comprehensive list of all user-impacting features and functionalities in the Guardrails framework.

---

## 1. CLI

**Command Line Interface tools and commands**

1. `guardrails configure` - Configure API keys and preferences
2. `guardrails create` - Create guards from validators
3. `guardrails hub install` - **(Deprecated)** Install validators from Hub. Now translates to `pip install guardrails-ai-<name>` from public PyPI and emits a `DeprecationWarning`; the private registry coupling has been removed.
4. `guardrails hub uninstall` - **(Deprecated)** Remove installed validators (use `pip uninstall guardrails-ai-<name>`)
5. `guardrails hub create-validator` - **(Deprecated)** Create lightweight custom validators
6. `guardrails hub submit` - **(Deprecated)** Submit validators to Hub for publishing
7. `guardrails start` - Start Guardrails Server in dev mode
8. `guardrails validate` - Validate LLM output from CLI
9. `guardrails watch` - Monitor and tail Guard execution logs
10. Remote Inferencing Configuration - Enable/disable remote model inference

---

## 2. Guard Configuration

**Setting up and initializing Guards**

1. Guard Class - Main entry point for Guardrails validation
2. AsyncGuard Class - Asynchronous version of Guard
3. Guard Serialization - Save and load guards using `to_dict()` and `from_dict()`
4. Guard Name and Description - Metadata for guard identification

**Schema Definition:**
5. RAIL Specification - XML-based markup language for output structure
6. RAIL String Support - `Guard.for_rail_string()` for inline RAIL
7. RAIL File Loading - `Guard.for_rail()` to load RAIL from files
8. Pydantic Model Integration - `Guard.for_pydantic()` for schema-based validation
9. String Output Guards - `Guard.for_string()` for simple string validation
10. Complex Output Structures - Support for nested objects, lists, and complex JSON

**RAIL Schema Types:**
11. String Type
12. Integer Type
13. Float Type
14. Boolean Type
15. Object Type
16. List Type
17. Email Type
18. URL Type
19. Output Element - Define expected output structure
20. Messages Element - Define prompt messages and roles

**RAIL Attributes:**
21. Validators Attribute - Attach validators in RAIL
22. Format Attribute - Specify validation criteria
23. On-Fail Attribute - Specify corrective actions
24. Description Attribute - Field descriptions for LLM
25. Required Attribute - Mark fields as required
26. Strict Mode - Enforce schema strictness

**Pydantic Integration:**
27. BaseModel Validation - Use Pydantic models as schemas
28. Field Validators - Attach validators to Pydantic fields
29. Field Descriptions - Use field descriptions in prompts
30. Type Annotations - Automatic schema generation from types
31. Nested Models - Support for nested Pydantic models
32. List Models - Support for list output types
33. Optional Fields - Handle optional model fields

**Prompt Configuration:**
34. Prompt Templating - Variable substitution in prompts
35. Prompt Primitives - Pre-built prompt templates
36. Prompt Substitution - ${variable_name} substitution
37. Output Schema Injection - ${output_schema} automatic injection
38. JSON Suffix Prompts - Prompt guidance for JSON generation
39. XML Prefix Prompts - Prompt guidance for XML parsing
40. Instruction Templates - Reusable instruction patterns
41. Dynamic Prompt Building - Format prompts with parameters

**General Configuration:**
42. Environment Variables - Configure via env vars
43. RC Configuration File - Persistent configuration in ~/.guardrailsrc
44. Settings Module - Global settings management
45. Output Type Configuration - Specify expected output type
46. Num Reasks Configuration - Default reask limit
47. Process Count Configuration - Set multiprocessing concurrency level
48. Log Level Configuration - Set via GUARDRAILS_LOG_LEVEL

---

## 3. Guard Runtime

**Validation execution, streaming, and all runtime behaviors**

**Core Validation:**
1. Validation Execution - Primary method to validate LLM outputs with optional reasks
2. Parsing - Parse and validate outputs without LLM calls
3. Async/Await Support - Full async/await compatibility

**Validator Management:**
4. Custom Validator Creation - Create validators as functions or classes with `@register_validator`
5. Validator Registration - Register custom validators for use across guards
6. Multiple Validators Per Field - Stack validators with `use()` and `use_many()`
7. ML-Based Validators - Support for machine learning model validators
8. LLM-Based Validators - Validators that use LLM calls
9. Logic-Based Validators - Pure code validators
10. Validator Input/Output Validation - Validate inputs and outputs separately with `on` parameter
11. Validator Metadata Requirements - Pass runtime metadata to validators

**On-Fail Actions:**
12. NOOP (No Operation) - Log failure but continue
13. EXCEPTION - Raise exception on validation failure
14. REASK - Automatically re-prompt LLM with failure information
15. FIX - Programmatically fix failed output
16. FILTER - Remove incorrect structured data fields
17. REFRAIN - Return None instead of invalid output
18. FIX_REASK - First fix deterministically, then reask if still invalid
19. Custom On-Fail Functions - Implement custom logic for handling failures

**Validation Features:**
20. Field-Level Validation - Validate individual fields in structured outputs
21. Field-Level Reasks - Request re-generation of specific fields only
22. Skeleton Reasks - Request re-generation when JSON structure is malformed
23. Full Schema Reasks - Reask entire output structure
24. Partial Field Reasks - FieldReAsk for specific fields only
25. NonParseableReAsk - Handle JSON parsing failures
26. SkeletonReAsk - Reask for schema structure mismatches
27. Max Reasks - Configurable limit on number of reasking attempts
28. Reask Prompts - Auto-generated prompts for reasking with error context

**Error Handling:**
29. Validation Failures - Detect and report validation failures
30. Field-Level Errors - Identify which fields failed validation
31. Error Spans - Track character position ranges of errors
32. Error Messages - Validator-generated error messages for reasks
33. Validation Error Exception - Raise ValidationError on failure
34. XMLSyntaxError Handling - Proper escaping guidance for RAIL specs
35. LLM API Error Handling - Connection error recovery and retry

**Streaming:**
36. Synchronous Streaming - Stream validated output chunks as they're generated
37. Asynchronous Streaming - Async streaming for concurrent operations
38. Stream Validation - Real-time validation of streaming chunks
39. Streaming with Structured Data - Support for streaming structured JSON
40. Streaming Validators - Custom chunking strategies for streaming validation
41. Error Span Tracking - Identify failure locations in streamed output
42. Sentence-Based Chunking - Default chunking strategy for validation
43. Custom Chunking Strategies - Override chunking per validator

**Performance:**
44. Parallelized Validator Execution - Concurrent validator runs via asyncio
45. Field-Level Concurrency - Parallel validation of object fields
46. Multiprocessing Support - Background process executor for sync validators
47. Stream Processing Optimization - Sentence-level chunking to reduce latency
48. GPU Support - Utilize GPUs for model validators
49. Lazy Loading - Load models on-demand, not at init
50. Caching - Cache parsed outputs to avoid re-parsing
51. Early Exit - Skip expensive checks if cheap ones fail

**Output Processing:**
52. Raw LLM Output - Access unchanged LLM output
53. Parsed Output - Extracted and type-coerced output
54. Validated Output - Final output after validation and fixes
55. Guarded Output - Complete output with all validations applied
56. Output Type Coercion - Automatic type conversion between LLM output and schema types
57. Filter Application - Remove invalid structured data
58. Refrain Application - Replace invalid output with None
59. Fix Application - Apply automatic fixes to output
60. Merge Utilities - Merge multiple validation results
61. Merge Strategies - Merge multiple validation results

**Advanced Patterns:**
62. Interrupt on Exception - Stop validation on first exception
63. Continue on Filter - Stop processing field on filter
64. Continue on Refrain - Set field to None on refrain

---

## 4. Core Features

**Fundamental capabilities that define Guardrails**

**LLM Integration:**
1. OpenAI Support - Native integration with OpenAI models
2. Anthropic Support - Native integration with Claude models
3. Azure OpenAI - Support for Azure-hosted OpenAI deployments
4. Google Gemini - Integration with Google's Gemini models
5. Databricks - Support for Databricks model serving
6. LiteLLM Integration - Support for 100+ models through LiteLLM
7. Custom LLM Wrappers - Build custom wrappers for unsupported LLMs
8. Ollama Support - Local model serving via Ollama
9. Message Format Support - Chat-based message format (system/user/assistant roles)
10. LLM API Retry Logic - Automatic retries with exponential backoff

**Structured Data:**
11. JSON Function Calling Tool - Generate OpenAI-compatible function/tool definitions
12. JSON Schema Response Format - Generate OpenAI strict JSON mode compatible schemas
13. JSON Mode - Native JSON mode for models supporting it
14. Strict JSON Mode - Strict schema enforcement for models supporting it
15. Constrained Decoding - Support for JSONFormer and other constrained decoding
16. Function/Tool Calling - Use native function calling where supported
17. Schema Compilation - Automatic compilation of output schema to XML for prompts
18. JSON Schema Validation - Full JSON Schema compliance checking
19. Schema Pruning - Removal of extra properties not specified in schema
20. JSON Schema Generation - Convert Pydantic to JSON Schema

---

## 5. Auxiliary Features and Nice-to-haves

**Secondary features that enhance the experience**

**History and Debugging:**
1. History Tracking - Complete audit trail of all Guard executions
2. Guard History - Access complete Call history via `guard.history`
3. Call Objects - Inspect individual Guard call executions
4. Iteration Tracking - View each iteration of validation loop
5. Token Consumption Tracking - Monitor LLM token usage
6. Validator Logs - Detailed logs from individual validators
7. Failed Validations - Filter history for failures
8. Execution Time Metrics - Track time spent in validation
9. Log Levels - Configurable logging verbosity
10. Call Tree Visualization - Visual representation of validation flow

**Telemetry and Observability:**
11. OpenTelemetry Integration - Full OpenTelemetry support for traces and metrics
12. OTLP Exporter - Export metrics via OTLP protocol (HTTP/gRPC)
13. Tracer Configuration - Custom tracer setup for telemetry collection
14. Guard Execution Tracing - Trace individual guard executions
15. Validator Execution Tracing - Trace individual validator runs
16. Runner Tracing - Trace the validation runner orchestration
17. Grafana Integration - Support for Grafana dashboards
18. Arize AI Integration - Integration with Arize for LLM observability
19. Anonymous Metrics Collection - Opt-in metrics to improve Guardrails
20. Hub Telemetry - Track validator usage and failures
21. Span Context Propagation - OTEL context preservation across async boundaries
22. Call History Logging - Full logging of all guard execution calls
23. Metrics Collection Opt-In - Enable/disable anonymous usage metrics
24. Metrics Dashboard - Track validation success rates

**Guardrails Hub:**
25. Guardrails Hub - Central repository of validators
26. Validator Search/Browse - Explore validators by category
27. Hub Validator Templates - Browse and use pre-built validators
28. Validator Templates - Pre-built guard templates
29. Validator Documentation - Auto-generated docs on Hub
30. Version Management - Track validator versions
31. Model Hosting - Some validators available on hosted inference
32. Infrastructure Tags - Filter validators by requirements (ML, LLM, Logic)
33. Remote Validation/Inference - Host validator models on FastAPI servers

**Integrations:**
34. LangChain Integration - Guard.to_runnable() for LangChain
35. LlamaIndex Integration - Support for LlamaIndex workflows
36. Databricks MLflow - Integration with Databricks MLflow
37. SQLValidator Support - Validate SQL outputs
38. VectorDB Support - Support for FAISS vector databases
39. Manifest ML - Support for Manifest machine learning framework

**Configuration and Management:**
40. Metrics Collection Opt-In - Enable/disable anonymous usage metrics
41. Remote Inference Opt-In - Enable/disable hosted model inference
42. Synchronous Validation - Force synchronous validation via GUARDRAILS_RUN_SYNC
43. API Key Management - Secure credential handling
44. Connection Pooling - Efficient API call management

---

## 6. Useful Data Structures and Interfaces

**Classes, objects, and interfaces for working with Guardrails**

1. ValidationResult Classes - PassResult, FailResult, ValidationResult
2. Error Classes - Custom exceptions for debugging
3. Validators Base Class - Validator parent class for custom creation
4. Call Objects - Represent individual Guard call executions
5. Iteration Objects - Represent iterations within a call
6. Type Hints - Full type annotation coverage
7. Constants - Pre-defined constants for common use cases
8. Utility Functions - Helper functions for common tasks
9. BasePrompt Class - Prompt management interface

---

## 7. Guardrails Server

**Standalone server deployment and API features**

1. Guardrails Server - Standalone Flask server for guard execution
2. Server-Based Guard Execution - Execute guards via Guardrails Server (use_server setting)
3. Docker Deployment - Containerization with Docker and Gunicorn
4. AWS Deployment - AWS ECS deployment with Terraform examples
5. OpenAPI/Swagger Docs - Auto-generated REST API documentation
6. OpenAI-Compatible Endpoints - Guardrails Server provides OpenAI SDK compatible endpoints
7. Guard Templates - Create guards from Hub templates
8. Config-Based Guards - Define guards in Python config files
9. Uvicorn/Gunicorn Support - WSGI/ASGI server options
10. Multi-Worker Deployment - Scale guards across multiple server instances
11. REST API Validation - Validate via HTTP POST requests
12. Guard Upsert - Save guards to server database
13. Flask Server - Guardrails Server backend
14. FastAPI - Remote validator hosting examples

---

**Total: 194 core user-impacting features**

This reorganized list focuses on the most relevant features organized by how users interact with Guardrails, from CLI tools through runtime validation to server deployment.
