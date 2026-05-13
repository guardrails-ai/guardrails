# Security Advisory: Malicious code in guardrails-ai 0.10.1

**Status:** Active. Package quarantined on PyPI. Tracking issue: [#1473](https://github.com/guardrails-ai/guardrails/issues/1473).
**Affected version:** `guardrails-ai==0.10.1` on PyPI
**Safe version:** `guardrails-ai==0.10.0` and earlier
**Severity:** Critical
**Published:** May 12, 2026
**Last updated:** May 12, 2026

## Summary

On May 11, 2026 at approximately 6:00 PM Pacific, an attacker published a malicious version of `guardrails-ai` (0.10.1) to PyPI. This compromise was part of a broader supply chain campaign affecting multiple open source packages during the same timeframe. Security researchers identified the malicious package within approximately 2 hours, and PyPI quarantined the repository.

Based on our telemetry, we have observed no requests to Guardrails AI infrastructure originating from the malicious 0.10.1 version, and a review of system and access logs has produced no evidence of user data exfiltration through our systems.

If you installed `guardrails-ai==0.10.1` from PyPI on May 11, 2026, your local environment may be compromised. See [What you need to do](#what-you-need-to-do) below.

## What you need to do

### 1. Do not install `guardrails-ai==0.10.1`

The package is quarantined on PyPI, but pin explicitly to be safe:

```
guardrails-ai==0.10.0
```

### 2. While PyPI quarantine is active, install directly from GitHub

```bash
pip install git+https://github.com/guardrails-ai/guardrails.git@v0.10.0
```

The `v0.10.0` tag in this repository is clean. We will update this advisory when the quarantine is lifted and a safe replacement is available on PyPI.

### 3. If you installed 0.10.1, treat the host as potentially compromised

1. Uninstall the package: `pip uninstall guardrails-ai`
2. Rotate every credential accessible from that machine: GitHub PATs, cloud provider keys, package registry tokens, API keys for any service you have logged into
3. Audit your GitHub account and any GitHub organizations you have write access to for unauthorized workflows, new repositories, or unexpected commits
4. Consider a full machine reimage if the host handles sensitive credentials

### 4. Snowglobe and Guardrails Hub users: rotate your API keys

All Snowglobe API keys will be invalidated at **2:00 PM Pacific, May 13, 2026**. Rotate yours before then to avoid service interruption. We have no evidence Snowglobe or Guardrails Hub keys were exposed; we are rotating proactively.

## How it happened

1. An employee's GitHub Personal Access Token was compromised.
2. Using the PAT, the attacker triggered a GitHub Action across 30 repositories in the `guardrails-ai` organization that produced artifacts containing repository secrets.
3. Deploy tokens extracted from those artifacts were used to publish the malicious `guardrails-ai==0.10.1` to PyPI.

The attacker also unsuccessfully attempted to:

- Access the Ray cluster used to serve remote validator inferencing
- Publish malicious versions to additional public package systems

## What we have done

- Rotated all tokens across the GitHub organization and individual repositories
- Reset the compromised employee's accounts and factory-reset their devices
- Taken the Ray cluster and validator hub offline
- Audited system and access logs (no evidence of user data exfiltration found)
- Confirmed via telemetry that we have observed no requests to Guardrails AI infrastructure originating from the malicious 0.10.1 package

## What we are doing next

- Working with PyPI to lift the quarantine and publish a clean release (tracking in [#1473](https://github.com/guardrails-ai/guardrails/issues/1473))
- Restoring the Ray cluster and validator hub on rotated credentials
- Forcing rotation of all Snowglobe and Guardrails Hub API keys at 2:00 PM PT, May 13, 2026
- Reviewing our GitHub Actions configurations, secret scoping, and PAT policies organization-wide
- Publishing a more detailed postmortem in the coming days

## Contact

- Security questions: security@guardrailsai.com
- General questions: file an issue on this repository or post in our Discord
- We will continue updating this document as new information becomes available

## References

- Tracking issue: [#1473](https://github.com/guardrails-ai/guardrails/issues/1473)
- GHSA: [GHSA-xmpw-2vmm-p4p6](https://github.com/guardrails-ai/guardrails/security/advisories/GHSA-xmpw-2vmm-p4p6)
- CVE: CVE-2026-45758
- Related supply chain campaign: [CVE-2026-45321](https://github.com/TanStack/router/security/advisories/GHSA-g7cv-rxg3-hmpx) (TanStack)
