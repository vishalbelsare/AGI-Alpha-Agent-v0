# α‑AGI Insight Demo — v0

This demo illustrates a minimal **Meta‑Agentic Tree Search** setup that
autonomously searches for sectors likely to experience AGI disruption. It
runs entirely offline and requires zero data.

```bash
python -m alpha_factory_v1.demos.alpha_agi_insight_v0.insight_demo --episodes 5
```

The search loop mutates a candidate sector index and evaluates it with a
lightweight number‑line environment. After a few iterations the tree returns
the sector with the best score.
