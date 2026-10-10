"""Run the Jev (and optionally Claude) benchmark from the command line and export the results.

Examples
  python scripts/run_benchmark.py --demo                    # simulated Jev, no keys needed
  python scripts/run_benchmark.py --provider both --limit 100 --live
  python scripts/run_benchmark.py --provider jev --out results/
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--provider", choices=["jev", "claude", "both"], default="jev")
    ap.add_argument("--limit", type=int, default=None, help="stratified subset size (default: all cases)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--workers", type=int, default=None)
    mode = ap.add_mutually_exclusive_group()
    mode.add_argument("--demo", action="store_true", help="force Demo Mode (simulated Jev, no API keys)")
    mode.add_argument("--live", action="store_true", help="force live mode (needs TYPESAFE_API_KEY)")
    ap.add_argument("--out", default=str(ROOT / "results"), help="output directory for CSV/JSON exports")
    args = ap.parse_args()

    if args.demo:
        os.environ["DENTOROUTE_DEMO_MODE"] = "true"
    elif args.live:
        os.environ["DENTOROUTE_DEMO_MODE"] = "false"

    from backend.services import benchmark_service, export_service
    from backend.utils.config import get_settings
    from backend.utils.errors import DentoRouteError

    demo = get_settings().resolve_demo_mode()
    print(f"Mode: {'DEMO (simulated Jev, not real model output)' if demo else 'LIVE'}")

    def progress(done: int, total: int) -> None:
        if done % 25 == 0 or done == total:
            print(f"  {done}/{total} cases", end="\r", flush=True)

    try:
        out = benchmark_service.run_benchmark(args.provider, args.limit, None, args.workers, args.seed, progress)
    except DentoRouteError as exc:
        print(f"\nBenchmark could not run: {exc.user_message}", file=sys.stderr)
        return 2
    print()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    for provider, run in out["runs"].items():
        tag = f"{provider}_{run['mode']}"
        (out_dir / f"benchmark_{tag}.csv").write_text(export_service.rows_to_csv(run["rows"]), encoding="utf-8")
        (out_dir / f"benchmark_{tag}.json").write_text(export_service.benchmark_json(run), encoding="utf-8")
        m = run["summary"]["metrics"]
        sim = " [SIMULATED]" if run["summary"]["simulated"] else ""
        print(f"\n== {provider} ({run['model']}){sim}: {run['n_cases']} cases, {run['n_errors']} errors, "
              f"schema compliance {m['schema_compliance']:.1%}")
        for q, v in m["per_question"].items():
            print(f"   {q:18s} accuracy {v['accuracy']:.3f}" if v["accuracy"] is not None else f"   {q:18s} n/a")
        print(f"   latency: {m['latency'].get('p50_ms', 'not measured')}  cost: {m['cost']['status']}")
        cal = run["summary"]["calibration"]["pooled"]
        print(f"   pooled ECE: {cal['ece']}  | {cal['confidence']['answer']}")
    for provider, why in out["skipped"].items():
        print(f"\n{provider}: {why}")
    if out["comparison"]:
        (out_dir / "benchmark_comparison.json").write_text(json.dumps(out["comparison"], indent=2), encoding="utf-8")
        print("\nPaired comparison (Jev - Claude accuracy, bootstrap 95% CI):")
        for q, v in out["comparison"]["paired"]["per_question"].items():
            print(f"   {q:18s} {v['difference']:+.3f}  CI [{v['ci95'][0]:+.3f}, {v['ci95'][1]:+.3f}]  {v['verdict']}")
    print(f"\nExports written to {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
