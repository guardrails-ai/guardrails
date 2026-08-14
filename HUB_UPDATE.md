**tl;dr**: Guardrails validators are moving to standard PyPI packages you install directly with `pip`, and Guardrails is discontinuing its hosted remote inferencing. See **How to Migrate** for what to do. **Hard cutoff: August 6, 2026.**

# Why
The Guardrails Hub has been the standard way to install AI guardrails for LLM engineers since 2024. With `guardrails hub install`, you could pull down dozens of validators to secure your agent inputs and outputs.

This pattern had a few main advantages:
1. Guardrails AI was able to curate third-party packages and offer them through the hub (albeit without guarantees).
2. Post-install actions could be orchestrated through the CLI, like installing models and auto-wiring remote inferencing.

However, the drawbacks now outweigh these benefits. The main feedback we've heard from engineers and platform teams is that guardrails validators do not install cleanly via `pip` or `uv`. Additionally, hosted remote inferencing has been offered for free since 2024, and the cost of running it is high.

Moving to plain PyPI packages makes Guardrails easier to install, maintain, iterate on, and contribute to.

# What happens on August 6, 2026 (hard cutoff)
On this date, two things are shut down:

1. **`guardrails hub install` and the private validator registry (`pypi.guardrailsai.com`).** After this, validators install **only** from public PyPI (`pip install guardrails-ai-<name>`). Existing `guardrails hub install …` commands, and anything else relying on the private registry, will stop working.
2. **The hosted remote-inference servers (`hub.api.guardrailsai.com`).** Validators that ran their models on Guardrails' servers must switch to running the model locally or on your own hosted endpoint.

Please migrate before this date. Follow this issue for progress.

# How to Migrate

## 1. Install with `pip` instead of `guardrails hub install`
Each validator is now a standalone PyPI package named `guardrails-ai-<name>` (underscores become dashes):

```diff
- guardrails hub install hub://guardrails/detect_pii
+ pip install guardrails-ai-detect-pii
```

No `guardrails configure` / API key is required anymore — these are public packages. **50 of the 64 validators are available on PyPI today**, and the rest are landing over the coming weeks. Browse the validator catalog at https://www.guardrailsai.com/hub.

Update your imports to the `guardrails_ai` namespace. The **registered validator name is unchanged** (`guardrails/detect_pii`), so `Guard().use(...)` and any `format="guardrails/detect_pii"` in RAIL keep working — only the import path changes:

```diff
- from guardrails.hub import DetectPII
+ from guardrails_ai.detect_pii import DetectPII
```

## 2. Move off hosted remote inference
Some validators (e.g. `detect_pii`, `toxic_language`, `competitor_check`, `nsfw_text`) could run their models on Guardrails' hosted inference servers, which are shut down on August 6, 2026 (see the cutoff above). You have two options:

1. **Run the model locally** — pass `use_local=True` when constructing the validator (the model loads/downloads on your machine):
   ```python
   guard = Guard().use(DetectPII(use_local=True))
   ```
2. **Host your own inference endpoint** and point the validator at it with `validation_endpoint=...`. Guide: https://www.guardrailsai.com/docs/concepts/remote_validation_inference

As of the latest release, constructing one of these validators against the Guardrails-hosted endpoint emits a `DeprecationWarning` noting this date.