# Policy Runbook

This runbook defines safety controls for Alpha-Factory. All deployments must follow these
procedures.

## Sandboxing

Generated code snippets execute in a restricted subprocess. The environment variables
`SANDBOX_CPU_SEC` and `SANDBOX_MEM_MB` limit CPU time and memory. When `firejail` is
available, snippets run under `firejail --net=none --private` to block network access.

## Timeouts

Sandboxed code is terminated when it exceeds the CPU limit. The orchestrator aborts any
simulation step that runs longer than the configured timeout to avoid runaway loops.

## Human Review

Policy changes and new agents require human code review before promotion. Features are
verified in a staging environment with tests and manual inspection.

## Rollback

`git tag stable` tracks the last known good release. If a new deployment fails, check out
the tag and redeploy the previous container or wheel. Investigate the issue before
retagging a fixed commit as `stable`.

## Promotion Checklist for Self‑Modifying Code

1. `pre-commit run --all-files`
2. `python check_env.py --auto-install` and `pytest -q`
3. Confirm sandbox limits are set in `.env` or the deployment manifest
4. Obtain approval from two maintainers
5. Update `docs/CHANGELOG.md` and create a signed tag
