# Handling Fix Results for Streaming in Guardrails

## Overview

This document describes how Guardrails handles fix results for streaming, addressing the challenges that arise from the new streaming architecture implemented to reduce latency and cost of validation.

## Fix Actions

Fix actions are a feature in Guardrails that allow you to specify an action to take when a particular validator fails. Some validators support a "FIX" on_fail action, which programmatically corrects faulty LLM output.

### Examples

1. Detect PII Validator:

   ```
   Input:  John lives in San Francisco.
   Output: <PERSON> lives in <LOCATION>
   ```

2. Lowercase Validator:
   ```
   Input:  JOHN lives IN san FRANCISCO.
   Output: john lives in san francisco.
   ```

In non-streaming scenarios, fix results from one validator can be piped into subsequent validators.

## Challenges with Streaming

The new streaming architecture allows validators to specify how much context to accumulate before running validation. However, this poses challenges for fix actions:

1. Validators cannot run in sequence due to different chunk thresholds.
2. Each validator accumulates chunks independently.
3. Validators are unaware of fixes applied by other validators.

## Solution: Merging Algorithm

To address these challenges, Guardrails implements the following solution:

1. Wait for all validators to accumulate enough chunks to validate and output a fix value.
2. Run a merging algorithm on the fixes from all validators.

### Merging Process

The merging algorithm combines fix outputs from multiple validators. For example:

```
LLM Output:        JOE is FUNNY and LIVES in NEW york
PII Fix Output:    <PERSON> is FUNNY and lives in <LOCATION>
Lowercase Fix:     joe is funny and lives in new york
Merged Output:     <PERSON> is funny and lives in <LOCATION>
```

### Implementation Details

- The merging algorithm is a modified version of the `three-merge` package.
- It uses Google's `diff-match-patch` algorithm under the hood.

## Limitations and Edge Cases

While the merging algorithm works well for most cases, there are some limitations:

- Edge cases may occur when replacement ranges overlap between multiple validators.
- In rare instances, the merging algorithm might produce unexpected results.

## Reporting Issues

If you encounter any bugs related to stream fixes, please:

1. File an issue on the [Guardrails GitHub repository](https://github.com/guardrails-ai/guardrails).
2. Mention @nichwch in the issue description.
