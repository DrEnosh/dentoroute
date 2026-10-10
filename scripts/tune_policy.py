"""Select routing-gate thresholds on train + calibration data, then compare old vs revised policy on the holdout.

  python scripts/tune_policy.py --demo            # run a simulated Jev benchmark first (no keys)
  python scripts/tune_policy.py --live            # run a live Jev benchmark first (needs TYPESAFE_API_KEY)
  python scripts/tune_policy.py                   # reuse the latest stored Jev benchmark
  python scripts/tune_policy.py --demo --write    # also write the selected policy to config/routing_policy.yaml

The holdout split is never used to fit, select or compare candidates. Thresholds are research-demo values,
not clinical thresholds. Without --write the active policy is not changed.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def pct(x: float | None) -> str:
    return "n/a" if x is None else f"{x:6.1%}"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    mode = ap.add_mutually_exclusive_group()
    mode.add_argument("--demo", action="store_true", help="run a simulated benchmark (all cases) first")
    mode.add_argument("--live", action="store_true", help="run a live Jev benchmark (all cases) first")
    ap.add_argument("--write", action="store_true", help="write the selected policy to config/routing_policy.yaml")
    ap.add_argument("--out", default=str(ROOT / "results"))
    args = ap.parse_args()
    if args.demo:
        os.environ["DENTOROUTE_DEMO_MODE"] = "true"
    elif args.live:
        os.environ["DENTOROUTE_DEMO_MODE"] = "false"

    from backend.services import benchmark_service, policy_eval_service as pe
    from backend.services.policy_service import dump_policy
    from backend.utils.config import CONFIG_DIR, reset_policy_cache
    from backend.utils.errors import DentoRouteError

    try:
        if args.demo or args.live:
            benchmark_service.run_benchmark("jev", None, demo=args.demo)
        study = pe.run_and_store_study()
    except DentoRouteError as exc:
        print(f"Could not run the study: {exc.user_message}", file=sys.stderr)
        return 2

    sel, prov = study["selection"], study["provenance"]
    print(f"Data: {prov['data_mode'].upper()} ({prov['model']}){'  [SIMULATED - not Jev]' if study['simulated'] else ''}")
    print(f"Splits (seed {prov['split_seed']}): {prov['split_counts']}  tuned on {prov['tuned_on']}, holdout untouched")
    print(f"Candidates: {sel['n_candidates']}, meeting the missed-urgent budget on calibration: {sel['n_feasible']}")
    if sel["note"]:
        print("NOTE:", sel["note"])
    print("Selected thresholds (after safety margin):", json.dumps(sel["selected_final"]))
    print(f"\nHOLDOUT ({study['per_split']['holdout']['legacy']['n']} cases, "
          f"{study['per_split']['holdout']['legacy']['n_urgent_reference']} reference-urgent)")
    print(f"{'metric':36s} {'legacy':>10s} {'revised':>10s}")
    h = study["per_split"]["holdout"]
    rows = [
        ("missed urgent (automated)", "missed_urgent", False), ("  upper 95% bound on missed rate", "missed_urgent_upper95", True),
        ("false automation (needs human)", "false_automation", False), ("unnecessary human review", "unnecessary_human_review", False),
        ("coverage (automated share)", "coverage", True), ("automation accuracy: specialty", "automation_accuracy_specialty", True),
        ("automation accuracy: urgency", "automation_accuracy_urgency", True), ("expected cost per case", "cost_per_case", None),
    ]
    for label, key, is_pct in rows:
        a, b = h["legacy"][key], h["revised"][key]
        f = (lambda v: pct(v)) if is_pct else (lambda v: f"{v:10.2f}" if isinstance(v, float) else f"{v:>10}")
        print(f"{label:36s} {f(a):>10s} {f(b):>10s}")
    pc = study["paired_holdout"]
    print(f"\nPaired bootstrap (holdout, revised - legacy): cost/case {pc['cost_per_case']['mean_difference_new_minus_old']:+.2f} "
          f"CI {[round(x, 2) for x in pc['cost_per_case']['ci95']]}; coverage {pc['coverage']['mean_difference_new_minus_old']:+.1%} "
          f"CI {[round(x, 3) for x in pc['coverage']['ci95']]}")
    print("Cost-ratio sensitivity (holdout cost per case):")
    for s in study["cost_sensitivity_holdout"]:
        print(f"  {s['missed_to_unnecessary_cost_ratio']:>4}:1  legacy {s['legacy_cost_per_case']:.2f}  revised {s['revised_cost_per_case']:.2f}")

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "policy_study.json").write_text(json.dumps(pe.public_study(study), indent=2), encoding="utf-8")
    print(f"\nStudy saved to {out / 'policy_study.json'}")
    if args.write:
        target = CONFIG_DIR / "routing_policy.yaml"
        header = "\n".join(l for l in target.read_text(encoding="utf-8").splitlines() if l.startswith("#"))
        dump_policy(pe.policy_for_writing(study), target, header)
        reset_policy_cache()
        print(f"Wrote {target}")
    else:
        print("Active policy NOT changed (use --write to apply).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
