# α‑AGI Insight Demo — v0

The **α‑AGI Insight** demo predicts which industry sector is most likely to be
transformed by Artificial General Intelligence.  It runs a small
**Meta‑Agentic Tree Search** (MATS) over a list of sector names.  No external
data is required so the script executes fully offline.

```bash
python -m alpha_factory_v1.demos.alpha_agi_insight_v0.insight_demo --episodes 5
```

## Usage

The command line interface mirrors the options of the general MATS demo:

```bash
python -m alpha_factory_v1.demos.alpha_agi_insight_v0.insight_demo \
    --episodes 10 \
    --rewriter openai \
    --model gpt-4o \
    --log-dir logs
```

When optional dependencies such as ``openai`` or ``anthropic`` are not
installed, the program automatically falls back to a simple offline rewriter so
the demo remains functional anywhere.  Episode scores are printed to the console
and optionally written to ``scores.csv`` when ``--log-dir`` is supplied.
