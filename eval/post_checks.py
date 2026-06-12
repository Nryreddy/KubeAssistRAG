"""
Eval Post Checks Module.

Contains deterministic evaluation functions that run after the LLM generates an answer.
Includes checks for forbidden keywords (security/PII) and source document overlap.
"""
from __future__ import annotations

def forbidden_keywords_check(answer: str, forbidden: list[str]) -> dict:
    answer_lower = answer.lower()
    hits = [kw for kw in forbidden if kw.lower() in answer_lower]
    return {"passed": not hits, "hits": hits}

def source_overlap(actual: list[str], golden: list[str]) -> dict:
    import os

    def _norm(s: str) -> str:
        return os.path.splitext(os.path.basename(s.strip().lower()))[0]

    actual_set = {_norm(s) for s in actual}
    golden_set = {_norm(s) for s in golden}
    overlap = actual_set & golden_set

    return {
        "overlap_pct": round(len(overlap) / max(len(golden_set), 1), 3),
        "matched": sorted(overlap),
        "missed": sorted(golden_set - actual_set),
    }