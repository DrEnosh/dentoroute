"""Regenerate the synthetic dataset + ground truth + demo cases (deterministic for a given seed).

    python scripts/generate_dataset.py [--per-category 25] [--seed 42]
"""
from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from backend.services import synthetic_case_service as svc  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-category", type=int, default=svc.DEFAULT_PER_CATEGORY)
    ap.add_argument("--seed", type=int, default=svc.DEFAULT_SEED)
    args = ap.parse_args()

    cases = svc.generate_dataset(args.per_category, args.seed)
    svc.save_dataset(cases)
    svc.save_demo_cases()
    print(f"Wrote {len(cases)} synthetic cases (seed={args.seed}) + ground truth + demo cases")
    for key in ("urgency", "specialty", "complexity", "referral_priority"):
        print(f"  {key:18s}", dict(Counter(getattr(c.reference, key) for c in cases)))
    for key in ("infection_concern", "same_day_review", "human_review"):
        print(f"  {key:18s}", dict(Counter(getattr(c.reference, key) for c in cases)))


if __name__ == "__main__":
    main()
