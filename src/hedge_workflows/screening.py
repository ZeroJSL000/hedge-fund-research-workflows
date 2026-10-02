"""Separate product exposure from manager background, then fuse the signals."""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass
from typing import Mapping

from .evidence import Extraction, evidence_supports_extraction


PRODUCT_LABELS = frozenset(
    {
        "Ex-China",
        "Greater China",
        "China Direct",
        "Asia Watchlist",
        "HK/Taiwan Watchlist",
        "Likely Non-China",
        "Unknown",
    }
)
MANAGER_LABELS = frozenset(
    {"Mainland-linked", "HK China-linked", "HK-based Weak Signal", "Foreign", "Unknown"}
)
CONFIDENCE = frozenset({"High", "Medium", "Low"})

_EX_CHINA = re.compile(r"\b(?:ex[\s-]?china|excluding\s+china|exclude\s+china)\b", re.I)
_GREATER_CHINA = re.compile(r"\bgreater\s+china\b", re.I)
_CHINA_DIRECT = re.compile(r"\b(?:mainland\s+china|china\s+(?:equities|equity|bonds|markets?)|a[\s-]?shares?)\b", re.I)
_ASIA = re.compile(r"\b(?:asia|asian|apac|asia[\s-]?pacific)\b", re.I)
_HK_TW = re.compile(r"\b(?:hong\s+kong|taiwan|hkex)\b", re.I)
_NON_CHINA = re.compile(r"\b(?:north\s+america|europe|japan|australia)\b", re.I)
_MAINLAND_OWNER = re.compile(r"\b(?:mainland\s+(?:parent|owner|ownership)|china\s+state[\s-]?owned\s+parent)\b", re.I)
_HK_CHINA_OWNER = re.compile(r"\b(?:hong\s+kong\s+subsidiary\s+of\s+(?:a\s+)?mainland|mainland[\s-]?owned\s+hong\s+kong)\b", re.I)
_FOREIGN_OWNER = re.compile(r"\b(?:independent\s+non[\s-]?china\s+manager|foreign\s+parent)\b", re.I)


@dataclass(frozen=True)
class Fund:
    fund_id: str
    fund_name: str
    product_description: str
    geography: str
    benchmark: str
    manager_id: str
    manager_name: str
    manager_description: str
    manager_location: str
    aum_rank: int | None = None

    @classmethod
    def from_row(cls, row: Mapping[str, str]) -> "Fund":
        required = ("fund_id", "fund_name", "manager_id")
        missing = [key for key in required if not row.get(key, "").strip()]
        if missing:
            raise ValueError(f"Fund row is missing: {', '.join(missing)}")
        rank = row.get("aum_rank", "").strip()
        return cls(
            fund_id=row["fund_id"].strip(),
            fund_name=row["fund_name"].strip(),
            product_description=row.get("product_description", "").strip(),
            geography=row.get("geography", "").strip(),
            benchmark=row.get("benchmark", "").strip(),
            manager_id=row["manager_id"].strip(),
            manager_name=row.get("manager_name", "").strip(),
            manager_description=row.get("manager_description", "").strip(),
            manager_location=row.get("manager_location", "").strip(),
            aum_rank=int(rank) if rank else None,
        )


@dataclass(frozen=True)
class Signal:
    label: str
    confidence: str
    matched_terms: tuple[str, ...] = ()
    conflict: bool = False


@dataclass(frozen=True)
class ScreeningPolicy:
    """Only a demonstration rank cutoff; real review budgets need calibration."""

    review_top_rank: int = 20

    def __post_init__(self) -> None:
        if self.review_top_rank < 0:
            raise ValueError("review_top_rank cannot be negative")


@dataclass(frozen=True)
class ScreeningResult:
    fund_id: str
    product_exposure: str
    manager_background: str
    bucket: str
    confidence: str
    manual_priority: str
    reason: str
    product_evidence_urls: tuple[str, ...]
    manager_evidence_urls: tuple[str, ...]
    rule_product: str
    rule_manager: str

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def _hits(pattern: re.Pattern[str], text: str) -> tuple[str, ...]:
    return tuple(dict.fromkeys(match.group(0) for match in pattern.finditer(text)))


def classify_product(fund: Fund) -> Signal:
    """Read only fund product fields; manager ownership cannot prove exposure."""

    text = " | ".join((fund.fund_name, fund.product_description, fund.geography, fund.benchmark))
    exclusion = _hits(_EX_CHINA, text)
    greater = _hits(_GREATER_CHINA, text)
    direct = _hits(_CHINA_DIRECT, text)
    if exclusion:
        return Signal("Ex-China", "High", exclusion, conflict=bool(greater or direct))
    if greater:
        return Signal("Greater China", "High", greater)
    if direct:
        return Signal("China Direct", "High", direct)
    if _ASIA.search(text):
        return Signal("Asia Watchlist", "Medium", _hits(_ASIA, text))
    if _HK_TW.search(text):
        return Signal("HK/Taiwan Watchlist", "Medium", _hits(_HK_TW, text))
    if _NON_CHINA.search(text):
        return Signal("Likely Non-China", "Medium", _hits(_NON_CHINA, text))
    return Signal("Unknown", "Low")


def classify_manager(fund: Fund) -> Signal:
    """Require an explicit ownership clue for a strong manager classification."""

    text = fund.manager_description
    if _HK_CHINA_OWNER.search(text):
        return Signal("HK China-linked", "High", _hits(_HK_CHINA_OWNER, text))
    if _MAINLAND_OWNER.search(text):
        return Signal("Mainland-linked", "High", _hits(_MAINLAND_OWNER, text))
    if _FOREIGN_OWNER.search(text):
        return Signal("Foreign", "Medium", _hits(_FOREIGN_OWNER, text))
    if _HK_TW.search(fund.manager_location):
        return Signal("HK-based Weak Signal", "Low", _hits(_HK_TW, fund.manager_location))
    return Signal("Unknown", "Low")


def _accepted_extraction(
    extraction: Extraction | None, label_set: frozenset[str], evidence: list
) -> Extraction | None:
    if extraction is None or extraction.label not in label_set or extraction.confidence not in CONFIDENCE:
        return None
    if extraction.confidence == "Low" or not evidence_supports_extraction(extraction, evidence):
        return None
    return extraction


def fuse_signals(
    fund: Fund,
    product_extraction: Extraction | None = None,
    manager_extraction: Extraction | None = None,
    product_evidence: list | None = None,
    manager_evidence: list | None = None,
    policy: ScreeningPolicy | None = None,
) -> ScreeningResult:
    """Use cited extractions conservatively and route contradictions to review."""

    policy = policy or ScreeningPolicy()
    product_rule = classify_product(fund)
    manager_rule = classify_manager(fund)
    product_evidence = product_evidence or []
    manager_evidence = manager_evidence or []
    p_ext = _accepted_extraction(product_extraction, PRODUCT_LABELS, product_evidence)
    m_ext = _accepted_extraction(manager_extraction, MANAGER_LABELS, manager_evidence)

    product = product_rule.label
    manager = manager_rule.label
    product_conf = product_rule.confidence
    manager_conf = manager_rule.confidence
    conflict = product_rule.conflict
    if p_ext and p_ext.label != "Unknown":
        if product not in ("Unknown", p_ext.label) and product_conf == "High":
            conflict = True
        elif product == "Unknown" or p_ext.confidence == "High":
            product, product_conf = p_ext.label, p_ext.confidence
    if m_ext and m_ext.label != "Unknown":
        if manager not in ("Unknown", "HK-based Weak Signal", m_ext.label) and manager_conf == "High":
            conflict = True
        elif manager in ("Unknown", "HK-based Weak Signal") or m_ext.confidence == "High":
            manager, manager_conf = m_ext.label, m_ext.confidence

    linked = manager in ("Mainland-linked", "HK China-linked")
    if conflict or (product == "Ex-China" and linked):
        bucket, confidence, reason = "Need Manual Check", "Low", "Conflicting product or ownership signals"
    elif product in ("China Direct", "Greater China"):
        bucket, confidence, reason = "Confirmed China-related", product_conf, "Product evidence indicates direct exposure"
    elif product == "Ex-China":
        bucket, confidence, reason = "Likely Non-China", product_conf, "Product explicitly excludes China"
    elif linked:
        bucket, confidence, reason = "Potential China-related", manager_conf, "Manager link does not establish product exposure"
    elif product in ("Asia Watchlist", "HK/Taiwan Watchlist") or manager == "HK-based Weak Signal":
        bucket, confidence, reason = "Asia/HK Watchlist", "Medium", "Regional signal requires product-level verification"
    elif product == "Likely Non-China":
        bucket, confidence, reason = "Likely Non-China", product_conf, "Product fields point to another region"
    elif fund.aum_rank is not None and fund.aum_rank <= policy.review_top_rank:
        bucket, confidence, reason = "Need Manual Check", "Low", "High-ranked fund lacks sufficient evidence"
    else:
        bucket, confidence, reason = "Unknown Low Priority", "Low", "No decisive product or ownership evidence"

    if bucket == "Need Manual Check" and conflict:
        priority = "High"
    elif bucket == "Need Manual Check" or (bucket == "Potential China-related" and fund.aum_rank is not None and fund.aum_rank <= policy.review_top_rank):
        priority = "Medium"
    elif bucket in ("Potential China-related", "Asia/HK Watchlist"):
        priority = "Low"
    else:
        priority = "No"

    return ScreeningResult(
        fund_id=fund.fund_id,
        product_exposure=product,
        manager_background=manager,
        bucket=bucket,
        confidence=confidence,
        manual_priority=priority,
        reason=reason,
        product_evidence_urls=(p_ext.evidence_url,) if p_ext else (),
        manager_evidence_urls=(m_ext.evidence_url,) if m_ext else (),
        rule_product=product_rule.label,
        rule_manager=manager_rule.label,
    )
