# Inspecting logs

All `gd.Guard` calls are logged internally, and can be accessed via two methods, `gd.Guard.guard_state` or `guardrails.log`.


## 🪵 Accessing logs via `guardrails.log`

This is the simplest way to access logs. It returns a list of all `gd.Guard` calls, in the order they were made.

In order to access logs, run:

```bash

eliot-tree --output-format=ascii guardrails.log

```


## 🇻🇦 Accessing logs via `gd.Guard.guard_state`

`guard_state` is an attribute of the `gd.Guard` class. It contains:

1. A list of all `gd.Guard` calls, in the order they were made.
2. For each call, reasks needed and their results.


## ⏲️ Coming soon!

On the roadmap: first class support for reading logs.



