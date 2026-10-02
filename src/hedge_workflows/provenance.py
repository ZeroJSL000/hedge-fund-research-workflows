"""A compact evidence matrix with source and numerical comparability checks."""

from __future__ import annotations

import json
import math
from collections import defaultdict
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from urllib.parse import urlparse


@dataclass(frozen=True)
class Claim:
    claim_id: str
    claim_key: str
    claim: str
    source_url: str
    document_title: str
    page: int | None
    observation_date: str
    unit: str
    denominator: str
    value: float | None
    qualifier: str


def read_claims(path: Path) -> list[Claim]:
    claims: list[Claim] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            item = json.loads(line)
            claims.append(Claim(**item))
        except (TypeError, ValueError, json.JSONDecodeError) as exc:
            raise ValueError(f"Invalid claim at line {line_number}: {exc}") from exc
    return claims


def validate_claims(claims: list[Claim]) -> dict[str, object]:
    """Flag incomplete provenance and conflicting or incomparable measures.

    Two values are directly comparable only when claim key, observation date,
    unit, and denominator agree. This does not establish that either is true.
    """

    issues: list[dict[str, str]] = []
    seen_ids: set[str] = set()
    comparable: dict[tuple[str, str, str, str], list[Claim]] = defaultdict(list)
    bases: dict[tuple[str, str], set[tuple[str, str]]] = defaultdict(set)
    for row in claims:
        if not row.claim_id or row.claim_id in seen_ids:
            issues.append({"claim_id": row.claim_id, "type": "duplicate_or_missing_id"})
        seen_ids.add(row.claim_id)
        if not row.claim_key or not row.claim or not row.document_title:
            issues.append({"claim_id": row.claim_id, "type": "missing_claim_or_title"})
        parsed = urlparse(row.source_url)
        if parsed.scheme not in ("http", "https") or not parsed.netloc:
            issues.append({"claim_id": row.claim_id, "type": "invalid_source_url"})
        if row.page is not None and (not isinstance(row.page, int) or isinstance(row.page, bool) or row.page < 1):
            issues.append({"claim_id": row.claim_id, "type": "invalid_page"})
        try:
            date.fromisoformat(row.observation_date)
        except (TypeError, ValueError):
            issues.append({"claim_id": row.claim_id, "type": "invalid_observation_date"})
        if row.value is not None and (
            not isinstance(row.value, (int, float))
            or isinstance(row.value, bool)
            or not math.isfinite(float(row.value))
        ):
            issues.append({"claim_id": row.claim_id, "type": "invalid_numeric_value"})
            continue
        if row.value is not None and (not row.unit or not row.denominator):
            issues.append({"claim_id": row.claim_id, "type": "missing_numeric_basis"})
        if not row.qualifier:
            issues.append({"claim_id": row.claim_id, "type": "missing_qualifier"})
        if row.value is not None:
            comparable[(row.claim_key, row.observation_date, row.unit, row.denominator)].append(row)
            bases[(row.claim_key, row.observation_date)].add((row.unit, row.denominator))
    for rows in comparable.values():
        if len({round(float(row.value), 9) for row in rows if row.value is not None}) > 1:
            issues.append({"claim_id": ",".join(row.claim_id for row in rows), "type": "conflicting_values"})
    for key, units in bases.items():
        if len(units) > 1:
            issues.append({"claim_id": key[0], "type": "incomparable_unit_or_denominator"})
    return {
        "claim_count": len(claims),
        "source_count": len({row.source_url for row in claims}),
        "issues": issues,
        "issue_count": len(issues),
        "scope": "Structural audit only; every substantive claim still needs source review",
    }
