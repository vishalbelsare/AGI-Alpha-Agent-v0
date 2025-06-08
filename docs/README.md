# Project Documentation

## Building the React Dashboard

The React dashboard sources live under `src/interface/web_client`. Build the static assets before serving the API:

```bash
pnpm --dir src/interface/web_client install
pnpm --dir src/interface/web_client run build
```

The compiled files appear in `src/interface/web_client/dist` and are automatically served when running `uvicorn src.interface.api_server:app` with `RUN_MODE=web`.

## Ablation Runner

Use `src/tools/ablation_runner.py` to measure how disabling individual innovations affects benchmark performance. The script applies each patch from `benchmarks/patch_library/`, runs the benchmarks with and without each feature and generates `docs/ablation_heatmap.svg`.

```bash
python -m src.tools.ablation_runner
```

The resulting heatmap visualises the pass rate drop when a component is disabled.

## Manual Workflows

The repository defines several optional GitHub Actions that are disabled by
default. They only run when the repository owner starts them from the GitHub
UI. These workflows perform heavyweight benchmarking and stress testing.

To launch a job:

1. Open the **Actions** tab on GitHub.
2. Choose either **📈 Replay Bench**, **🌩 Load Test** or **📊 Transfer Matrix**.
3. Click **Run workflow** and confirm.

Each workflow checks that the person triggering it matches
`github.repository_owner`, so it executes only when the owner initiates the
run.
