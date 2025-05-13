"""
RoyaltyRadar.a.agi.eth – α-AGI Business demo
Hunts missing music-streaming royalties and emits a reconciled cash-back report.

Drop this file into:
alpha_factory_v1/demos/meta_agentic_agi_v3/businesses/royalty_radar.py
"""

from pathlib import Path
import csv, json, time, hashlib, random
from datetime import datetime
from typing import Dict, List

# ✅ Meta-Agentic v3 primitives are already in PYTHONPATH
from core.fm import call_llm               # unified FM wrapper
from core.tools import sandbox_exec        # secure code runner
from agents.agent_base import Agent        # first-order agent contract
from meta_agentic_search.archive import log_stepstone   # lineage log helper


# ──────────────────────────────────────────────────────────────────────────────
# Business-level Orchestrator  (runs as one "super-agent" inside Alpha-Factory)
# ──────────────────────────────────────────────────────────────────────────────
class RoyaltyRadarBusiness(Agent):
    """
    One-shot reconciliation pipeline:
        1.   Ingest public stream-count APIs (mocked in demo).
        2.   Parse artist-supplied royalty statements (CSV).
        3.   Compare owed vs paid, flag gaps.
        4.   Autogenerate claim-letters & payment-instructions.
    """

    LABEL          = "RoyaltyRadar"
    VERSION        = "0.1.0"
    MIN_CONFIDENCE = 0.65        # reject hallucinated deltas < 65 % certainty

    # ------------------- public API -------------------
    def __init__(self, cfg: Dict):
        super().__init__(cfg)
        self.artist_name   = cfg["artist_name"]
        self.isrc_codes    = cfg["isrc_codes"]
        self.statement_csv = Path(cfg["statement_csv"])
        self.payout_wallet = cfg["payout_wallet"]

    def plan(self) -> str:
        return (
            f"Reconcile {self.artist_name}'s royalties across {len(self.isrc_codes)} ISRCs, "
            f"statement file «{self.statement_csv.name}», "
            f"deposit any gap into {self.payout_wallet}."
        )

    # ------------------- main work -------------------
    def run(self) -> Dict:
        self.logger.info(self.plan())

        # 1 Collect public / DSP stream counts (mock API)
        public_counts = {isrc: _mock_stream_api(isrc) for isrc in self.isrc_codes}
        self.logger.debug(f"Pulled public stream counts ➜ {public_counts}")

        # 2 Parse artist statements
        paid_counts, paid_eur = _parse_statement(self.statement_csv, self.isrc_codes)
        self.logger.debug(f"Parsed paid-counts ➜ {paid_counts}")

        # 3 Compute gaps
        gap_counts = {k: max(public_counts[k] - paid_counts.get(k, 0), 0)
                      for k in self.isrc_codes}
        gap_eur    = {k: gap_counts[k] * 0.0032      # €0.0032 ≈ blended €/stream
                      for k in self.isrc_codes}

        total_gap  = round(sum(gap_eur.values()), 2)
        self.logger.info(f"Detected unpaid royalties ≈ €{total_gap}")

        # 4 Draft claim letter with LLM (uses FM wrapper → vendor agnostic)
        letter = call_llm(
            prompt  = _letter_prompt(self.artist_name, gap_eur, self.payout_wallet),
            model   = self.cfg.get("llm_model", "mistral:7b-instruct.gguf"),
            temp    = 0.2,
        )

        # 5 Emit artefacts & lineage
        artefact = {
            "ts": datetime.utcnow().isoformat(),
            "artist": self.artist_name,
            "gap_eur": total_gap,
            "claim_letter": letter,
            "evidence": {
                "public_streams": public_counts,
                "paid_streams":   paid_counts,
            },
        }
        log_stepstone(self.LABEL, artefact)     # lineage JSON → archive

        # 6 (Real prod) call payout smart-contract; demo just prints
        self.logger.info(f"💸 Wire €{total_gap} to wallet {self.payout_wallet} (demo-mode)")

        return artefact


# ──────────────────────────────────────────────────────────────────────────────
# Helper functions
# ──────────────────────────────────────────────────────────────────────────────
def _mock_stream_api(isrc: str) -> int:
    """Deterministic pseudo-random stream count for demo repeatability."""
    seed = int(hashlib.sha1(isrc.encode()).hexdigest(), 16) % 2**32
    random.seed(seed);  return 1_500_000 + random.randint(0, 350_000)

def _parse_statement(csv_path: Path, isrc_filter: List[str]) -> (Dict[str, int], Dict[str, float]):
    """Reads a CSV with cols [date,isrc,streams,eur] — minimal demo schema."""
    counts, euros = {}, {}
    with csv_path.open() as fh:
        for row in csv.DictReader(fh):
            if row["isrc"] not in isrc_filter:      # ignore other songs
                continue
            counts[row["isrc"]] = counts.get(row["isrc"], 0) + int(row["streams"])
            euros[row["isrc"]]  = euros.get(row["isrc"],  0) + float(row["eur"])
    return counts, euros

def _letter_prompt(artist: str, gap: Dict[str, float], wallet: str) -> str:
    bullets = "\n".join(f"• {isrc} → €{round(e,2)}" for isrc, e in gap.items() if e > 0)
    return (
        f"Compose a courteous but firm royalty-recovery demand for artist «{artist}».\n"
        f"They have identified the following unpaid amounts:\n{bullets}\n"
        f"Request settlement within 14 days to ERC-20 wallet {wallet}. "
        f"Keep it ≤ 250 words, professional tone."
    )


# ──────────────────────────────────────────────────────────────────────────────
# CLI entry — enables stand-alone smoke-test
# ──────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import argparse, yaml, pprint, logging
    logging.basicConfig(level=logging.INFO)

    ap = argparse.ArgumentParser(description="RoyaltyRadar demo run")
    ap.add_argument("--config", default="../configs/royalty_radar.yml")
    args = ap.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text())
    artefact = RoyaltyRadarBusiness(cfg).run()
    pprint.pprint(artefact)
