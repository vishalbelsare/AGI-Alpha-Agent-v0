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

### Quick Start Script

Run ``./run_insight_demo.sh`` from this directory for an instant launch. The
helper delegates to the package entry point so the demo works with or without
OpenAI API credentials.

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

Set the ``ALPHA_AGI_SECTORS`` environment variable to override the default
sector list without editing configuration files.  Provide a comma-separated
string or the path to a text file containing one sector per line.  The
``openai_agents_bridge`` respects the same variable when launching via the
Agents runtime so your custom sector lists work in both modes.

Export ``MCP_ENDPOINT`` to capture all prompts and replies for later audit using
the Model Context Protocol. When unset the logging step is silently skipped.

When optional dependencies such as ``openai`` or ``anthropic`` are not
installed, the program automatically falls back to a simple offline rewriter so
the demo remains functional anywhere.  Episode scores are printed to the console
and optionally written to ``scores.csv`` when ``--log-dir`` is supplied.

### Graceful Offline Mode

The demo automatically falls back to an offline search strategy whenever the
required API keys are absent or network access is restricted. Use
``--verify-env`` for a quick dependency check before launching.

## OpenAI Agents Bridge

Launch ``openai_agents_bridge.py`` to control the demo via the
`openai-agents` runtime and optionally the Google ADK A2A protocol:

```bash
python -m alpha_factory_v1.demos.alpha_agi_insight_v0.openai_agents_bridge --verify-env
```
The bridge automatically falls back to offline mode when the optional
packages or API keys are missing. Use ``--enable-adk`` to expose the agent via
the optional Google ADK gateway when available.

### MCP Logging

Set the ``MCP_ENDPOINT`` environment variable to automatically store all
LLM prompts and replies using the
[Model Context Protocol](https://www.anthropic.com/news/model-context-protocol).
This best-effort persistence operates transparently and never blocks the
search loop.
