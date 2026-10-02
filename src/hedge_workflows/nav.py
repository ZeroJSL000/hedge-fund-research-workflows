"""Plan batched observations and measure raw NAV availability without filling gaps."""

from __future__ import annotations

import csv
import math
from collections import defaultdict
from dataclasses import asdict, dataclass
from datetime import date, timedelta
from pathlib import Path
from typing import Iterable


@dataclass(frozen=True)
class Observation:
    ticker: str
    day: date
    nav: float
    batch_number: str | None = None


@dataclass(frozen=True)
class RequestWindow:
    batch_number: int
    tickers: tuple[str, ...]
    start: date
    end: date


@dataclass(frozen=True)
class CoveragePolicy:
    """Illustrative thresholds; provide research-specific values for a real study."""

    fresh_days: int = 45
    stale_days: int = 120
    good_month_ratio: float = 0.75
    usable_month_ratio: float = 0.40
    good_min_months: int = 6
    usable_min_months: int = 3

    def __post_init__(self) -> None:
        if self.fresh_days < 0 or self.stale_days < self.fresh_days:
            raise ValueError("Require 0 <= fresh_days <= stale_days")
        if not 0 <= self.usable_month_ratio <= self.good_month_ratio <= 1:
            raise ValueError("Require 0 <= usable ratio <= good ratio <= 1")
        if self.usable_min_months < 0 or self.good_min_months < self.usable_min_months:
            raise ValueError("Require 0 <= usable months <= good months")


@dataclass(frozen=True)
class CoverageResult:
    ticker: str
    observation_count: int
    unique_nav_months: int
    expected_months: int
    monthly_coverage_ratio: float
    unique_nav_quarters: int
    expected_quarters: int
    quarterly_coverage_ratio: float
    trailing_12m_months_with_nav: int
    trailing_12m_coverage_ratio: float
    last_nav_date: str
    days_since_last_nav: int | None
    max_gap_days: int | None
    duplicate_date_count: int
    non_positive_nav_count: int
    status: str


def normalize_ticker(value: str) -> str:
    """Normalize spacing and case while retaining the instrument suffix."""

    return " ".join(value.strip().upper().split())


def _add_months(day: date, months: int) -> date:
    year = day.year + (day.month - 1 + months) // 12
    month = (day.month - 1 + months) % 12 + 1
    next_month = date(year + (month == 12), month % 12 + 1, 1)
    last_day = (next_month - timedelta(days=1)).day
    return date(year, month, min(day.day, last_day))


def plan_requests(
    tickers: Iterable[str], start: date, end: date, *, batch_size: int = 25, window_months: int = 12
) -> list[RequestWindow]:
    """Create deterministic, inclusive date windows and ticker batches.

    This is a request plan only. The package has no licensed data service client.
    """

    if start > end:
        raise ValueError("start must not follow end")
    if batch_size < 1 or window_months < 1:
        raise ValueError("batch_size and window_months must be positive")
    unique = tuple(dict.fromkeys(normalize_ticker(t) for t in tickers if t.strip()))
    windows: list[tuple[date, date]] = []
    cursor = start
    while cursor <= end:
        stop = min(end, _add_months(cursor, window_months) - timedelta(days=1))
        windows.append((cursor, stop))
        cursor = stop + timedelta(days=1)
    return [
        RequestWindow(i // batch_size + 1, unique[i : i + batch_size], a, b)
        for i in range(0, len(unique), batch_size)
        for a, b in windows
    ]


def read_tickers(path: Path) -> list[str]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames or "ticker" not in reader.fieldnames:
            raise ValueError("Fund CSV must include a ticker column")
        return [row["ticker"] for row in reader if row.get("ticker")]


def read_observations(path: Path) -> list[Observation]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames or not {"ticker", "date", "nav"}.issubset(reader.fieldnames):
            raise ValueError("NAV CSV must include ticker,date,nav columns")
        rows: list[Observation] = []
        for line_number, row in enumerate(reader, 2):
            if not row["ticker"] or not row["date"] or not row["nav"]:
                continue
            try:
                value = float(row["nav"])
                if not math.isfinite(value):
                    raise ValueError("NAV must be finite")
                rows.append(
                    Observation(
                        normalize_ticker(row["ticker"]),
                        date.fromisoformat(row["date"]),
                        value,
                        row.get("batch_number") or None,
                    )
                )
            except ValueError as exc:
                raise ValueError(f"Invalid NAV CSV row {line_number}: {exc}") from exc
        return rows


def _month_count(start: date, end: date) -> int:
    return (end.year - start.year) * 12 + end.month - start.month + 1


def _quarter_count(start: date, end: date) -> int:
    return (end.year - start.year) * 4 + (end.month - 1) // 3 - (start.month - 1) // 3 + 1


def assess_coverage(
    tickers: Iterable[str],
    observations: Iterable[Observation],
    start: date,
    end: date,
    policy: CoveragePolicy | None = None,
) -> tuple[list[CoverageResult], dict[str, object]]:
    """Classify missing, stale, sparse, usable, and good raw series.

    Calendar months with no observation remain missing. Observation values are not
    interpolated or forward filled. Out-of-window rows are excluded from metrics.
    """

    if start > end:
        raise ValueError("start must not follow end")
    policy = policy or CoveragePolicy()
    universe = tuple(dict.fromkeys(normalize_ticker(t) for t in tickers if t.strip()))
    known = set(universe)
    grouped: dict[str, list[Observation]] = defaultdict(list)
    unmatched: set[str] = set()
    batch_tickers: dict[str, set[str]] = defaultdict(set)
    batch_rows: dict[str, int] = defaultdict(int)
    for item in observations:
        ticker = normalize_ticker(item.ticker)
        if item.batch_number and start <= item.day <= end:
            batch_tickers[item.batch_number].add(ticker)
            batch_rows[item.batch_number] += 1
        if ticker not in known:
            unmatched.add(ticker)
        elif start <= item.day <= end:
            grouped[ticker].append(item)

    expected = _month_count(start, end)
    expected_quarters = _quarter_count(start, end)
    trailing_start = max(start, _add_months(end.replace(day=1), -11))
    trailing_expected = _month_count(trailing_start, end)
    results: list[CoverageResult] = []
    for ticker in universe:
        rows = sorted(grouped[ticker], key=lambda x: x.day)
        days = sorted({item.day for item in rows})
        months = {(day.year, day.month) for day in days}
        quarters = {(day.year, (day.month - 1) // 3 + 1) for day in days}
        trailing_months = {(day.year, day.month) for day in days if day >= trailing_start}
        coverage = len(months) / expected
        lag = (end - days[-1]).days if days else None
        gaps = [(b - a).days for a, b in zip(days, days[1:])]
        if not rows:
            status = "Missing"
        elif lag is not None and lag > policy.stale_days:
            status = "Stale"
        elif (
            lag is not None
            and lag <= policy.fresh_days
            and coverage >= policy.good_month_ratio
            and len(months) >= policy.good_min_months
        ):
            status = "Good"
        elif (
            lag is not None
            and lag <= policy.stale_days
            and coverage >= policy.usable_month_ratio
            and len(months) >= policy.usable_min_months
        ):
            status = "Usable"
        else:
            status = "Sparse"
        results.append(
            CoverageResult(
                ticker=ticker,
                observation_count=len(rows),
                unique_nav_months=len(months),
                expected_months=expected,
                monthly_coverage_ratio=round(coverage, 6),
                unique_nav_quarters=len(quarters),
                expected_quarters=expected_quarters,
                quarterly_coverage_ratio=round(len(quarters) / expected_quarters, 6),
                trailing_12m_months_with_nav=len(trailing_months),
                trailing_12m_coverage_ratio=round(len(trailing_months) / trailing_expected, 6),
                last_nav_date=days[-1].isoformat() if days else "",
                days_since_last_nav=lag,
                max_gap_days=max(gaps) if gaps else None,
                duplicate_date_count=len(rows) - len(days),
                non_positive_nav_count=sum(item.nav <= 0 for item in rows),
                status=status,
            )
        )
    audit = {
        "fund_count": len(universe),
        "raw_observation_count_in_window": sum(r.observation_count for r in results),
        "matched_fund_count": sum(r.observation_count > 0 for r in results),
        "funds_without_nav": [r.ticker for r in results if r.observation_count == 0],
        "unmatched_tickers": sorted(unmatched),
        "batch_audit": [
            {
                "batch_number": batch,
                "observation_count": batch_rows[batch],
                "unique_tickers": len(tickers_in_batch),
                "matched_tickers": len(tickers_in_batch & known),
                "unmatched_tickers": sorted(tickers_in_batch - known),
            }
            for batch, tickers_in_batch in sorted(batch_tickers.items())
        ],
        "status_counts": {status: sum(r.status == status for r in results) for status in ("Good", "Usable", "Sparse", "Stale", "Missing")},
        "policy": asdict(policy),
        "analysis_start": start.isoformat(),
        "analysis_end": end.isoformat(),
        "no_forward_fill": True,
    }
    return results, audit
