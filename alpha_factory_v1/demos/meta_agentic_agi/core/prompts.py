"""prompts.py – Prompt & Template Registry (v0.2.0)
=====================================================
This module centralises **all** natural‑language system / user prompts used
by the Meta‑Agentic α‑AGI demo.  It provides:

• 🔖  *Versioned* prompt strings (with rich docstrings & provenance).
• 🪄  Helper builders that inject run‑time context (archive, examples, etc.).
• 🧩  A lightweight registry so new prompt variants can be composed &
       A/B‑tested by the search loop (multi‑objective optimisation ready).

The API is intentionally dependency‑free so it can be imported from any
component – including distributed workers – without triggering heavyweight
loads (e.g. OpenAI SDK).

Apache‑2.0 © 2025 MONTREAL.AI
"""

from __future__ import annotations

import json, inspect, hashlib, datetime as _dt
from pathlib import Path
from typing import List, Dict, Any

__all__ = [  # Public API
    "Prompt",
    "registry",
    "get",
    "register",
    "get_prompt",
    "get_init_archive",
    "get_reflexion_prompt",
]


# ---------------------------------------------------------------------------
# Core dataclass‑like container
# ---------------------------------------------------------------------------
class Prompt(Dict[str, str]):
    """Tiny helper so mypy understands we expect at least the four keys."""

    thought: str
    name: str
    code: str

    def __init__(self, *, thought: str, name: str, code: str) -> None:
        super().__init__(thought=thought.strip(), name=name.strip(), code=code.strip())

    # fingerprint for dedup / lineage UI
    @property
    def sha1(self) -> str:
        m = hashlib.sha1()
        m.update(self["code"].encode())
        return m.hexdigest()[:8]


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------
_REGISTRY: Dict[str, Prompt] = {}


def register(p: Prompt) -> None:
    if p["name"] in _REGISTRY:
        raise ValueError(f"prompt name already registered: {p['name']}")
    _REGISTRY[p["name"]] = p


def registry() -> Dict[str, Prompt]:
    """Return *copy* of registry (immutability outside)."""
    return dict(_REGISTRY)


def get(name: str) -> Prompt:
    return _REGISTRY[name]


# ---------------------------------------------------------------------------
# Canonical system / base blocks
# ---------------------------------------------------------------------------
SYSTEM_PROMPT = """You are a helpful assistant."""

BASE_TEMPLATE = Path(__file__).with_suffix(".base.md").read_text() if Path(__file__).with_suffix(".base.md").exists() else """    # Overview
You are an expert machine‑learning researcher...
[BASE TEMPLATE PLACEHOLDER – override by shipping a *.base.md file next to this
module for production deployments.]
"""

# ---------------------------------------------------------------------------
# Convenience builders (back‑compat with ADAS)
# ---------------------------------------------------------------------------
def get_prompt(current_archive: List[Prompt]) -> tuple[str, str]:
    archive_str = ",\n".join(json.dumps(sol, ensure_ascii=False) for sol in current_archive)
    prompt = BASE_TEMPLATE.replace("[ARCHIVE]", f"[{archive_str}]").replace(
        "[EXAMPLE]", json.dumps(EXAMPLE_AGENT, ensure_ascii=False)
    )
    return SYSTEM_PROMPT, prompt


def get_init_archive() -> List[Prompt]:
    return [COT_CODE, REFLEXION, LLM_DEBATE, COT_SC, QD]


def _prev_agent_str(p: Prompt | None) -> str:
    if not p:
        return ""
    return "Here is the previous agent you tried:\n" + json.dumps(p, ensure_ascii=False) + "\n\n"


def get_reflexion_prompt(prev: Prompt | None) -> tuple[str, str]:
    text1 = REFLEXION_PROMPT_1.replace("[EXAMPLE]", _prev_agent_str(prev))
    return text1, REFLEXION_PROMPT_2


# ---------------------------------------------------------------------------
# Seed agents (trimmed for brevity – full code lives in EXTERNAL_SEEDS/*.py)
# ---------------------------------------------------------------------------
# NOTE:  real implementations are imported lazily from disk to keep this file
#        lightweight.  Each seed agent has a peer *.py in the same folder.
def _load_seed(fname: str) -> str:
    path = Path(__file__).with_suffix("").with_name(fname)
    return path.read_text() if path.exists() else f"# Placeholder – missing {fname}"


COT_CODE = Prompt(
    thought="""Chain‑of‑thought with code generation.""".strip(),
    name="Chain‑of‑Thought",
    code=_load_seed("seed_cot_code.py"),
)

COT_SC = Prompt(
    thought="Self‑Consistency ensemble of Chain‑of‑Thought agents.",
    name="Self‑Consistency with CoT",
    code=_load_seed("seed_cot_sc.py"),
)

REFLEXION = Prompt(
    thought="Self‑Refine loop using explicit feedback (Reflexion).", name="Reflexion", code=_load_seed("seed_reflexion.py"),
)

LLM_DEBATE = Prompt(
    thought="Debate between role‑specialised LLMs (Game Designer vs Logician).", name="LLM Debate", code=_load_seed("seed_debate.py"),
)

QD = Prompt(
    thought="Quality‑Diversity search: generate N diverse solutions then ensemble.",
    name="Quality‑Diversity",
    code=_load_seed("seed_qd.py"),
)

# Example placeholder agent for docs
EXAMPLE_AGENT = Prompt(
    thought="Placeholder example.",
    name="Example Agent",
    code="def forward(self, taskInfo):\n    return []  # TODO"
)

# Register seeds
for _p in (COT_CODE, COT_SC, REFLEXION, LLM_DEBATE, QD):
    register(_p)

# ---------------------------------------------------------------------------
# House‑keeping
# ---------------------------------------------------------------------------
def _dump_registry_for_ui(path: str | Path = "registry_snapshot.json") -> None:
    """Utility: write registry to disk so the Lineage UI can render prompts."""
    Path(path).write_text(json.dumps(_REGISTRY, indent=2, ensure_ascii=False))

if __name__ == "__main__":
    # Simple smoke test
    import pprint

    print("# Registered prompts", len(_REGISTRY))
    pprint.pp(_REGISTRY)
    print("# get_prompt output snippet:\n", get_prompt(get_init_archive())[1][:750], "...\n")