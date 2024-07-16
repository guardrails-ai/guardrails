# Guardrails

Guardrails are a set of ML models or rules that are used to validate the output of a language model. They are used to ensure that the output of the model is safe, accurate, and meets the requirements of the user. Guardrails can be used to check for things like bias, toxicity, and other issues that may arise from using a language model.

Each guardrail validates the presence of a specific type of risk – that risk may range from unsafe code, hallucinations, regulation violations, company disrepute, toxicity or unsatisfactory user experience. Some examples of guardrails on the hub are:
- Anti-Hallucination guardrails
- No mentions of Competitors (for an organization)
- No toxic language
- Accurate summarization
- PII Leakage
- For code-gen use cases — generating invalid / unsafe code