"""Comparable-sample construction and manager-level performance summaries.

The quality tiers mirror the workflow structure used in the research project.
All cutoffs are supplied by the caller; this public module contains no private
screening thresholds, fund identities, or manager rankings.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import date, datetime
from math import isfinite, isnan
from statistics import median
from typing import Any, Iterable, Mapping


CORE_FIELDS = (
    "return_1y",
    "return_3y",
    "sharpe_3y",
    "volatility_3y",
    "max_drawdown",
    "last_nav_date",
)
RETURN_FIELDS = ("return_1y", "return_3y", "return_5y")


@dataclass(frozen=True)
class PerformancePolicy:
    """Illustrative, caller-defined screening and ranking parameters."""

    as_of: date
    stale_after_days: int
    numeric_bounds: Mapping[str, tuple[float, float]]
    minimum_manager_funds: int

    def __post_init__(self) -> None:
        if self.stale_after_days < 0:
            raise ValueError("stale_after_days must be nonnegative")
        if self.minimum_manager_funds < 1:
            raise ValueError("minimum_manager_funds must be positive")
        for field, (low, high) in self.numeric_bounds.items():
            if low > high:
                raise ValueError(f"Invalid numeric bounds for {field}")


@dataclass(frozen=True)
class QualityAssessment:
    """An exclusive quality tier with reasons retained for review."""

    tier: str
    reasons: tuple[str, ...]

    @property
    def comparable(self) -> bool:
        return self.tier in {"A1", "A2"}


def _present(value: Any) -> bool:
    if value is None:
        return False
    if isinstance(value, str):
        return value.strip().lower() not in {"", "na", "n/a", "nan", "null"}
    if isinstance(value, float):
        return not isnan(value)
    return True


def _number(value: Any) -> float | None:
    if not _present(value) or isinstance(value, bool):
        return None
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if isfinite(result) else None


def _date(value: Any) -> date | None:
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    if not _present(value):
        return None
    try:
        return date.fromisoformat(str(value)[:10])
    except ValueError:
        return None


def assess_quality(record: Mapping[str, Any], policy: PerformancePolicy) -> QualityAssessment:
    """Apply outlier, staleness, and completeness rules in that order."""

    reasons: list[str] = []
    for field, (low, high) in policy.numeric_bounds.items():
        raw = record.get(field)
        if not _present(raw):
            continue
        value = _number(raw)
        if value is None:
            reasons.append(f"{field}: unparseable")
        elif not low <= value <= high:
            reasons.append(f"{field}: outside configured bounds")
    if reasons:
        return QualityAssessment("F", tuple(reasons))

    last_nav = _date(record.get("last_nav_date"))
    if _present(record.get("last_nav_date")) and last_nav is None:
        return QualityAssessment("F", ("last_nav_date: unparseable",))
    if last_nav and last_nav > policy.as_of:
        return QualityAssessment("F", ("last_nav_date: after analysis date",))
    if last_nav and (policy.as_of - last_nav).days > policy.stale_after_days:
        return QualityAssessment("E", ("last NAV exceeds configured age",))

    core_complete = all(
        (last_nav is not None if field == "last_nav_date" else _number(record.get(field)) is not None)
        for field in CORE_FIELDS
    )
    if core_complete:
        extended_complete = _number(record.get("return_5y")) is not None and _number(record.get("aum_usd")) is not None
        return QualityAssessment("A2" if extended_complete else "A1", ())

    if _number(record.get("return_1y")) is not None and (
        _number(record.get("return_3y")) is None or _number(record.get("return_5y")) is None
    ):
        return QualityAssessment("B", ("short performance history",))

    if any(_number(record.get(field)) is not None for field in RETURN_FIELDS):
        return QualityAssessment("C", ("return available but required risk fields incomplete",))
    return QualityAssessment("D", ("insufficient performance disclosure",))


def build_comparable_sample(
    records: Iterable[Mapping[str, Any]], policy: PerformancePolicy
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Preserve all input records and record why each excluded row is excluded."""

    accepted: list[dict[str, Any]] = []
    excluded: list[dict[str, Any]] = []
    for source in records:
        row = dict(source)
        assessment = assess_quality(row, policy)
        row["quality_tier"] = assessment.tier
        row["quality_reasons"] = list(assessment.reasons)
        if not assessment.comparable:
            row["exclusion_reason"] = f"quality tier {assessment.tier}"
            excluded.append(row)
        elif not str(row.get("manager_name") or "").strip():
            row["exclusion_reason"] = "manager name missing"
            excluded.append(row)
        else:
            accepted.append(row)
    return accepted, excluded


def summarize_managers(
    comparable: Iterable[Mapping[str, Any]], policy: PerformancePolicy
) -> list[dict[str, Any]]:
    """Use fund-level medians and expose the sample-size reliability gate."""

    groups: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in comparable:
        manager = " ".join(str(row.get("manager_name") or "").split())
        if manager:
            groups[manager].append(row)

    summaries: list[dict[str, Any]] = []
    for manager, rows in groups.items():
        def median_field(field: str) -> float | None:
            values = [_number(row.get(field)) for row in rows]
            available = [value for value in values if value is not None]
            return median(available) if available else None

        strategies = Counter(str(row.get("strategy")) for row in rows if _present(row.get("strategy")))
        aum_values = [_number(row.get("aum_usd")) for row in rows]
        known_aum = [value for value in aum_values if value is not None]
        summaries.append(
            {
                "manager_name": manager,
                "comparable_fund_count": len(rows),
                "rank_eligible": len(rows) >= policy.minimum_manager_funds,
                "median_return_3y": median_field("return_3y"),
                "median_sharpe_3y": median_field("sharpe_3y"),
                "median_max_drawdown": median_field("max_drawdown"),
                "known_aum_usd": sum(known_aum) if known_aum else None,
                "aum_coverage": len(known_aum) / len(rows),
                "strategy_mix": dict(sorted(strategies.items())),
            }
        )
    return sorted(summaries, key=lambda row: row["manager_name"].casefold())


def rank_managers(summaries: Iterable[Mapping[str, Any]], metric: str) -> list[dict[str, Any]]:
    """Rank only managers passing the configured sample-size gate."""

    if metric not in {"median_return_3y", "median_sharpe_3y"}:
        raise ValueError("metric must be median_return_3y or median_sharpe_3y")
    eligible = [dict(row) for row in summaries if row.get("rank_eligible") and _number(row.get(metric)) is not None]
    eligible.sort(key=lambda row: (-float(row[metric]), str(row["manager_name"]).casefold()))
    for rank, row in enumerate(eligible, 1):
        row["rank"] = rank
    return eligible
