This repository is a conceptual research prototype. References to "AGI" and "superintelligence" describe aspirational goals and do not indicate the presence of a real general intelligence. Use at your own risk. Nothing herein constitutes financial advice. MontrealAI and the maintainers accept no liability for losses incurred from using this software.

# Sandbox Resource Limits

Generated code snippets run in a restricted subprocess. The following environment variables control the CPU time and memory available to the sandbox:

| Variable | Default | Description |
|----------|---------|-------------|
| `SANDBOX_CPU_SEC` | `2` | CPU time limit in seconds. |
| `SANDBOX_MEM_MB` | `256` | Maximum memory in megabytes. |

When unset, the defaults above are applied.
