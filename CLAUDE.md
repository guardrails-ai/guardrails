# Guardrails — contributor guidance for Claude

## Dependency strategy (IMPORTANT — always follow)

This repo separates **compatibility** (what users are allowed to install) from
**security** (what we actually ship and test against). The two live in different
files and must be managed differently:

- **`pyproject.toml` — keep dependency ranges WIDE.** Floors and ceilings here exist
  to express genuine compatibility boundaries (a real API break, a known-incompatible
  major), *not* to pin security-patched versions. Do not raise a floor just to dodge a
  CVE. Narrow ranges here propagate into every downstream project that installs
  `guardrails-ai`, causing resolver conflicts for users.
- **`poetry.lock` — pin the SAFE, patched versions.** Dependabot scans the lock file,
  so the lock is where security fixes land. Within the wide `pyproject` ranges, the lock
  should resolve to the latest patched releases.

### How to apply

- To fix a Dependabot/security alert: widen the `pyproject` range only as far as needed
  to *allow* the patched version (e.g. if the fix is in a new major, raise the ceiling),
  then update `poetry.lock` to the patched version with `poetry update <pkg>` (or a full
  `rm poetry.lock && poetry lock` to also re-resolve non-active marker branches —
  `poetry update` will not re-resolve marker branches like `python_version >= "3.x"`).
- Prefer fixing a transitive cap at its source: if a first-party package (e.g.
  `guardrails-api`) caps a dependency below a security fix, raise the cap in that package
  and release it, rather than pinning around it here.
- Only add an environment marker (e.g. `; python_version < '3.15'`) to a `pyproject`
  dependency when a transitive constraint makes a version genuinely unresolvable for part
  of the supported Python range — not as a convenience.

### Gates — all three must pass before any dependency change is "done"

```
make lint    # ruff check + format --check
make type    # pyright guardrails/
make test    # pytest tests/
```

Run them inside the project venv (`source .venv/bin/activate`). Never install into system
Python; if the venv is broken, recreate it (`make refresh`).
