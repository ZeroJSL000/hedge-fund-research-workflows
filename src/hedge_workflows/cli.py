"""Command line examples for the three public research workflow patterns."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from collections import Counter
from dataclasses import asdict
from datetime import date
from pathlib import Path

from .checkpoint import CheckpointStore, atomic_json_write, input_digest
from .evidence import OfflineEvidenceAdapter
from .nav import CoveragePolicy, assess_coverage, plan_requests, read_observations, read_tickers
from .performance import PerformancePolicy, build_comparable_sample, rank_managers, summarize_managers
from .provenance import read_claims, validate_claims
from .screening import Fund, ScreeningPolicy, fuse_signals


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError("Cannot write an empty CSV table")
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    try:
        temporary.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows), encoding="utf-8")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _run_nav(args: argparse.Namespace) -> None:
    start, end = date.fromisoformat(args.start), date.fromisoformat(args.end)
    tickers = read_tickers(args.funds)
    observations = read_observations(args.observations)
    policy = CoveragePolicy(
        fresh_days=args.fresh_days,
        stale_days=args.stale_days,
        good_month_ratio=args.good_month_ratio,
        usable_month_ratio=args.usable_month_ratio,
        good_min_months=args.good_min_months,
        usable_min_months=args.usable_min_months,
    )
    results, audit = assess_coverage(tickers, observations, start, end, policy)
    if not results:
        raise ValueError("Fund universe is empty")
    _write_csv(args.output, [asdict(row) for row in results])
    windows = plan_requests(tickers, start, end, batch_size=args.batch_size, window_months=args.window_months)
    manifest = {
        "workflow": "nav_coverage",
        "input_digest": input_digest(args.funds, args.observations),
        "audit": audit,
        "request_plan": [
            {"batch_number": w.batch_number, "tickers": list(w.tickers), "start": w.start.isoformat(), "end": w.end.isoformat()}
            for w in windows
        ],
        "note": "Request plan is illustrative; no external market-data request was made.",
    }
    atomic_json_write(args.output.with_suffix(".manifest.json"), manifest)
    print(json.dumps({"output": str(args.output), "status_counts": audit["status_counts"]}))


def _read_funds(path: Path) -> list[Fund]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames:
            raise ValueError("Fund CSV requires a header")
        funds = [Fund.from_row(row) for row in reader]
    if len({row.fund_id for row in funds}) != len(funds):
        raise ValueError("fund_id values must be unique")
    return funds


def _run_screen(args: argparse.Namespace) -> None:
    funds = _read_funds(args.funds)
    if not funds:
        raise ValueError("Fund CSV is empty")
    adapter = OfflineEvidenceAdapter(args.evidence)
    policy = ScreeningPolicy(review_top_rank=args.review_top_rank)
    digest = hashlib.sha256(
        (input_digest(args.funds, args.evidence) + f"|review_top_rank={policy.review_top_rank}").encode("utf-8")
    ).hexdigest()
    checkpoint = CheckpointStore(args.checkpoint, digest)
    for fund in funds:
        if checkpoint.contains(fund.fund_id):
            continue
        product_evidence = adapter.search("fund", fund.fund_id)
        manager_evidence = adapter.search("manager", fund.manager_id)
        product = adapter.extract("fund", fund.fund_id, product_evidence)
        manager = adapter.extract("manager", fund.manager_id, manager_evidence)
        result = fuse_signals(fund, product, manager, product_evidence, manager_evidence, policy)
        checkpoint.save_result(fund.fund_id, result.to_dict())
    rows = [checkpoint.results[fund.fund_id] for fund in funds]
    _write_jsonl(args.output, rows)
    counts = dict(Counter(str(row["bucket"]) for row in rows))
    atomic_json_write(
        args.output.with_suffix(".manifest.json"),
        {
            "workflow": "china_relevance_screening",
            "input_digest": digest,
            "fund_count": len(funds),
            "bucket_counts": counts,
            "manual_review_count": sum(row["manual_priority"] != "No" for row in rows),
            "method": "Separate product and manager labels; cited extraction; conservative rule fusion",
        },
    )
    print(json.dumps({"output": str(args.output), "bucket_counts": counts}))


def _run_provenance(args: argparse.Namespace) -> None:
    audit = validate_claims(read_claims(args.input))
    atomic_json_write(args.output, audit)
    print(json.dumps({"output": str(args.output), "issue_count": audit["issue_count"]}))


def _run_performance(args: argparse.Namespace) -> None:
    with args.records.open(newline="", encoding="utf-8-sig") as handle:
        records = list(csv.DictReader(handle))
    raw_policy = json.loads(args.policy.read_text(encoding="utf-8"))
    policy = PerformancePolicy(
        as_of=date.fromisoformat(raw_policy["as_of"]),
        stale_after_days=int(raw_policy["stale_after_days"]),
        numeric_bounds={key: tuple(bounds) for key, bounds in raw_policy["numeric_bounds"].items()},
        minimum_manager_funds=int(raw_policy["minimum_manager_funds"]),
    )
    accepted, excluded = build_comparable_sample(records, policy)
    summaries = summarize_managers(accepted, policy)
    result = {
        "workflow": "manager_performance",
        "input_digest": input_digest(args.records, args.policy),
        "accepted_count": len(accepted),
        "excluded_count": len(excluded),
        "accepted": accepted,
        "excluded": excluded,
        "manager_summaries": summaries,
        "rank_by_median_return_3y": rank_managers(summaries, "median_return_3y"),
        "rank_by_median_sharpe_3y": rank_managers(summaries, "median_sharpe_3y"),
    }
    atomic_json_write(args.output, result)
    print(json.dumps({"output": str(args.output), "accepted_count": len(accepted), "excluded_count": len(excluded)}))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="hedge-workflows", description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)

    nav = commands.add_parser("nav", help="Measure raw NAV availability from local CSV files")
    nav.add_argument("--funds", required=True, type=Path)
    nav.add_argument("--observations", required=True, type=Path)
    nav.add_argument("--start", required=True)
    nav.add_argument("--end", required=True)
    nav.add_argument("--output", required=True, type=Path)
    nav.add_argument("--batch-size", type=int, default=25)
    nav.add_argument("--window-months", type=int, default=12)
    nav.add_argument("--fresh-days", type=int, default=45)
    nav.add_argument("--stale-days", type=int, default=120)
    nav.add_argument("--good-month-ratio", type=float, default=0.75)
    nav.add_argument("--usable-month-ratio", type=float, default=0.40)
    nav.add_argument("--good-min-months", type=int, default=6)
    nav.add_argument("--usable-min-months", type=int, default=3)
    nav.set_defaults(run=_run_nav)

    screen = commands.add_parser("screen", help="Screen synthetic fund and manager evidence")
    screen.add_argument("--funds", required=True, type=Path)
    screen.add_argument("--evidence", required=True, type=Path)
    screen.add_argument("--output", required=True, type=Path)
    screen.add_argument("--checkpoint", required=True, type=Path)
    screen.add_argument("--review-top-rank", type=int, default=20)
    screen.set_defaults(run=_run_screen)

    provenance = commands.add_parser("provenance", help="Audit a synthetic claim evidence matrix")
    provenance.add_argument("--input", required=True, type=Path)
    provenance.add_argument("--output", required=True, type=Path)
    provenance.set_defaults(run=_run_provenance)

    performance = commands.add_parser("performance", help="Build a comparable fund sample and manager summaries")
    performance.add_argument("--records", required=True, type=Path)
    performance.add_argument("--policy", required=True, type=Path)
    performance.add_argument("--output", required=True, type=Path)
    performance.set_defaults(run=_run_performance)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    args.run(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
