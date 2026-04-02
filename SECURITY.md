# Security Policy

## Supported Versions

| Version | Supported |
|---------|-----------|
| 0.9.x   | Yes       |
| < 0.9   | No        |

## Reporting a Vulnerability

We take security seriously. If you find a vulnerability, please report it responsibly.

**Preferred method:** Open a report via [GitHub Security Advisories](https://github.com/guardrails-ai/guardrails/security/advisories/new). This keeps the details private until a fix ships.

**Fallback:** Email security@guardrailsai.com with a description of the issue, steps to reproduce, and any relevant logs or screenshots.

**Please do not** open a public GitHub issue for security vulnerabilities.

## Response Commitment

- **Acknowledge** your report within 48 hours.
- **Triage and assess** severity within 5 business days.
- **Ship a fix** within 7 days for critical vulnerabilities (CVSS 9.0+).
- **Coordinate disclosure** with the reporter before any public announcement.

## What Counts as a Security Issue

The following are in scope for security reports:

- **Supply chain attacks**: compromised dependencies, typosquatting, malicious packages in the dependency tree.
- **Credential exposure**: API keys, tokens, or secrets leaked through logs, error messages, or artifacts.
- **Remote code execution (RCE)**: any path that allows an attacker to execute arbitrary code on the host.
- **Sandbox escape**: bypassing validator isolation to access the host filesystem, network, or environment variables.
- **Injection attacks**: prompt injection, RAIL/XML injection, or other input manipulation that bypasses guards.

## Why This Policy Exists

The LiteLLM supply chain incident (compromised PyPI package with embedded credentials harvesting) demonstrated that LLM tooling libraries are high-value targets. Guardrails sits in the critical path between user input and LLM output, making supply chain integrity and dependency hygiene non-negotiable.

## Credit

Reporters who follow responsible disclosure will be credited by name (or handle) in the security advisory and the CHANGELOG entry for the fix release, unless they prefer to remain anonymous.
