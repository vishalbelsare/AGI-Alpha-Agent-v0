# α‑AGI Insight 👁️✨ — Beyond Human Foresight — Official Demo

The **α‑AGI Insight** demo predicts which industry sector is most likely to be
transformed by Artificial General Intelligence. It runs a small
**Meta‑Agentic Tree Search** (MATS) over a list of sector names. No external
 data is required so the script executes fully offline.  Pass a custom sector
 list with ``--sectors`` to experiment with your own domains. When the argument
 points to a text file, each non-empty line is treated as a sector name.

```bash
python -m alpha_factory_v1.demos.alpha_agi_insight_v0.insight_demo --episodes 5
```

### Single Command Launcher

Run the package itself to automatically select the best interface. By default it
routes through the OpenAI Agents runtime when available and transparently
degrades to the local CLI otherwise:

```bash
python -m alpha_factory_v1.demos.alpha_agi_insight_v0 --verify-env
```

Pass ``--offline`` to skip the agent runtime entirely.

## Usage

The command line interface mirrors the options of the general MATS demo:

```bash
python -m alpha_factory_v1.demos.alpha_agi_insight_v0.insight_demo \
    --episodes 10 \
    --rewriter openai \
    --model gpt-4o \
    --log-dir logs \
    --sectors Finance,Healthcare,Energy
```

Use ``--list-sectors`` to display the resolved sector list without running the
search. This is helpful when providing custom lists via ``--sectors``.

When optional dependencies such as ``openai`` or ``anthropic`` are not
installed, the program automatically falls back to a simple offline rewriter so
the demo remains functional anywhere.  Episode scores are printed to the console
and optionally written to ``scores.csv`` when ``--log-dir`` is supplied.

### Graceful Offline Mode

The demo detects missing API keys and transparently switches to a local search
strategy so it operates without network access. Supply ``--verify-env`` to run a
best‑effort dependency check before launching.

## OpenAI Agents Bridge

Launch ``openai_agents_bridge.py`` to control the demo via the
`openai-agents` runtime and optionally the Google ADK A2A protocol:

```bash
python -m alpha_factory_v1.demos.alpha_agi_insight_v0.openai_agents_bridge --verify-env
```
The bridge automatically falls back to offline mode when the optional
packages or API keys are missing. Use ``--enable-adk`` to expose the agent via
the optional Google ADK gateway when available.
