# ⚠️ DEPRECATED: `validator_pypi_publish`

**This action is deprecated and will be removed.**

It publishes a validator to the **private** Guardrails registry
(`pypi.guardrailsai.com`) via `twine`, using the legacy `<org>-grhub-<name>`
package-name mangling. Both the private registry and that naming scheme are
being retired.

## What replaces it

Guardrails-owned validators now live in the
[`guardrails-ai/guardrails-hub`](https://github.com/guardrails-ai/guardrails-hub)
monorepo and publish to **public PyPI** as `guardrails-ai-<name>` (importable as
`guardrails_ai.<name>`) using **trusted publishing** (OIDC — no tokens). See
`guardrails-hub/.github/workflows/validators_publish.yml`.

| Old | New |
|---|---|
| Private registry `pypi.guardrailsai.com` | Public PyPI |
| Dist name `guardrails-grhub-<name>` | `guardrails-ai-<name>` |
| `twine upload` with a Guardrails token | `pypa/gh-action-pypi-publish` via OIDC |
| This shared composite action | Change-detected `validators_publish.yml` in the monorepo |

## Why it's still here

The old per-validator repositories still reference this action at
`guardrails-ai/guardrails/.github/actions/validator_pypi_publish@main`. To avoid
breaking their CI, the action remains functional but now emits a loud
deprecation warning. **Full removal is tracked with the per-validator repo
archival phase** (out of scope for the current migration).
