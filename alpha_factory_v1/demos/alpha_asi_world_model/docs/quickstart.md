## Quick‑Start

## Disclaimer
This demo is a conceptual research prototype. References to "AGI" and
"superintelligence" describe aspirational goals and do not indicate the presence
of a real general intelligence. Use at your own risk.

```bash
pip install -r requirements.txt
python -m alpha_asi_world_model_demo --demo
```

### Wheelhouse Setup

Build wheels using the same Python version as your virtual environment and
verify packages from that directory:

```bash
mkdir -p /media/wheels
pip wheel -r requirements.txt -w /media/wheels
pip wheel -r ../../../requirements-dev.txt -w /media/wheels
python check_env.py --auto-install --wheelhouse /media/wheels
```

Set `WHEELHOUSE=/media/wheels` when launching the demo offline:

```bash
WHEELHOUSE=/media/wheels alpha-asi-demo --demo
```

### Running `check_env.py` offline

When network access is unavailable, pass `--wheelhouse <dir>` so `pip`
installs missing packages from your local cache instead of PyPI:

```bash
python check_env.py --auto-install --wheelhouse /media/wheels
```

Run `python scripts/check_python_deps.py` first; if it reports any
missing packages they **must** be installed from the wheelhouse using
the command above. Set `NO_LLM=1` when no `OPENAI_API_KEY` is supplied
to skip the external planner and run entirely with local models.
