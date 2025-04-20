import logging, json, random
from pathlib import Path
from .agent_base import AgentBase
from .model_provider import ModelProvider


_KB_DIR = Path("/var/alphafactory/policy_kb")
_KB_DIR.mkdir(parents=True, exist_ok=True)
_SAMPLE_DOC = _KB_DIR / "sample_policy.txt"
if not _SAMPLE_DOC.exists():
    _SAMPLE_DOC.write_text(
        "Climate Bill 2024: introduces a carbon fee of $50/ton beginning 2026.\n"
    )


def _retrieve_chunks(query: str, k: int = 3) -> str:
    # toy vector‑less retrieval: just return first k lines containing any keyword
    toks = set(query.lower().split())
    hits = []
    for p in _KB_DIR.glob("*.txt"):
        for line in p.read_text().splitlines():
            if toks & set(line.lower().split()):
                hits.append(line)
                if len(hits) >= k:
                    return "\n".join(hits)
    return "No relevant policy text found."


class PolicyAgent(AgentBase):
    def observe(self):
        # Triggered ad‑hoc by Planner; no external pull required
        return []

    def think(self, _obs):
        question = "What upcoming legislation could impact renewable‑energy investments?"
        context = _retrieve_chunks("renewable energy carbon")
        prompt = (
            f"Context:\n{context}\n\n"
            f"Question:\n{question}\n\n"
            "Answer briefly and cite the bill name."
        )
        answer = self.model.complete(prompt, max_tokens=200)
        idea = {"type": "insight", "question": question, "answer": answer}
        self.memory.write(self.name, "idea", idea)
        return [idea]

    def act(self, tasks):
        # For PoC: just log; future versions could push to Slack/email
        for t in tasks:
            self.memory.write(self.name, "action", t)
            self.log.info("Policy insight logged: %s", t["answer"])

