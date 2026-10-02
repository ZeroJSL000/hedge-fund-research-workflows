"""Provider interfaces and validation for cited search and JSON extraction."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol
from urllib.parse import urlparse


SOURCE_RANK = {
    "regulator": 5,
    "company_registry": 5,
    "official_site": 4,
    "factsheet": 4,
    "third_party_database": 3,
    "news": 2,
    "search_snippet": 1,
    "unknown": 0,
}
PRIMARY_SOURCE_TYPES = frozenset({"regulator", "company_registry", "official_site", "factsheet"})


@dataclass(frozen=True)
class Evidence:
    subject_type: str
    subject_id: str
    source_url: str
    source_type: str
    text: str
    observed_at: str

    def __post_init__(self) -> None:
        if self.subject_type not in ("fund", "manager"):
            raise ValueError("subject_type must be fund or manager")
        if self.source_type not in SOURCE_RANK:
            raise ValueError(f"Unknown source_type: {self.source_type}")
        parsed = urlparse(self.source_url)
        if parsed.scheme not in ("http", "https") or not parsed.netloc:
            raise ValueError("Evidence requires an HTTP(S) source URL")


@dataclass(frozen=True)
class Extraction:
    subject_type: str
    subject_id: str
    label: str
    confidence: str
    evidence_url: str
    evidence_text: str
    reason: str


class SearchProvider(Protocol):
    """Implement this interface around a licensed or public search service."""

    def search(self, subject_type: str, subject_id: str) -> list[Evidence]: ...


class JsonExtractor(Protocol):
    """Implement this interface around an LLM with strict JSON output."""

    def extract(self, subject_type: str, subject_id: str, evidence: list[Evidence]) -> Extraction | None: ...


def rank_evidence(rows: list[Evidence]) -> list[Evidence]:
    """Keep one record per URL and rank primary sources above search snippets."""

    deduped: dict[str, Evidence] = {}
    for row in rows:
        old = deduped.get(row.source_url)
        if old is None or (SOURCE_RANK[row.source_type], len(row.text)) > (SOURCE_RANK[old.source_type], len(old.text)):
            deduped[row.source_url] = row
    return sorted(deduped.values(), key=lambda row: (-SOURCE_RANK[row.source_type], row.source_url))


def evidence_supports_extraction(extraction: Extraction, evidence: list[Evidence]) -> bool:
    """Reject uncited claims, snippets with inflated confidence, and quote mismatches."""

    for row in evidence:
        if (row.subject_type, row.subject_id, row.source_url) != (
            extraction.subject_type,
            extraction.subject_id,
            extraction.evidence_url,
        ):
            continue
        quote = " ".join(extraction.evidence_text.casefold().split())
        body = " ".join(row.text.casefold().split())
        if not quote or quote not in body:
            return False
        if extraction.confidence == "High" and row.source_type not in PRIMARY_SOURCE_TYPES:
            return False
        if extraction.confidence == "Medium" and SOURCE_RANK[row.source_type] < 2:
            return False
        return True
    return False


def extraction_prompt(subject_type: str, evidence: list[Evidence]) -> dict[str, Any]:
    """Create provider-neutral instructions for a JSON-only extraction call.

    A caller can submit this dictionary to any model API. The package neither
    includes credentials nor sends data to an external service.
    """

    if subject_type not in ("fund", "manager"):
        raise ValueError("subject_type must be fund or manager")
    subject = "product geographic exposure" if subject_type == "fund" else "manager ownership background"
    return {
        "task": f"Extract {subject} from cited evidence",
        "rules": [
            "Use only the supplied evidence and return one JSON object.",
            "Do not equate a manager's background with a fund's investment exposure.",
            "Asia or Hong Kong location alone does not prove direct mainland exposure or ownership.",
            "An exclusion such as ex-China must be interpreted as an exclusion.",
            "Use Unknown with Low confidence when evidence is insufficient.",
            "Copy evidence_url and an exact supporting evidence_text quote from one supplied record.",
        ],
        "required_keys": ["label", "confidence", "evidence_url", "evidence_text", "reason"],
        "evidence": [
            {"source_url": row.source_url, "source_type": row.source_type, "text": row.text[:4000]}
            for row in rank_evidence(evidence)
        ],
    }


def parse_extraction(subject_type: str, subject_id: str, payload: dict[str, Any]) -> Extraction:
    """Validate the shape of a model response before applying research rules."""

    required = ("label", "confidence", "evidence_url", "evidence_text", "reason")
    if any(not isinstance(payload.get(key), str) for key in required):
        raise ValueError("Extraction must contain string label, confidence, evidence_url, evidence_text, reason")
    return Extraction(subject_type, subject_id, *(payload[key].strip() for key in required))


class OfflineEvidenceAdapter(SearchProvider, JsonExtractor):
    """Replay synthetic evidence and prevalidated JSON without network calls."""

    def __init__(self, path: Path):
        data = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            raise ValueError("Evidence file must contain a JSON object")
        self._evidence: dict[tuple[str, str], list[Evidence]] = {}
        self._extractions: dict[tuple[str, str], Extraction] = {}
        for item in data.get("evidence", []):
            row = Evidence(**item)
            self._evidence.setdefault((row.subject_type, row.subject_id), []).append(row)
        for item in data.get("extractions", []):
            if not isinstance(item, dict) or not isinstance(item.get("subject_type"), str) or not isinstance(item.get("subject_id"), str):
                raise ValueError("Extraction item requires subject_type and subject_id")
            subject_type, subject_id = item["subject_type"], item["subject_id"]
            self._extractions[(subject_type, subject_id)] = parse_extraction(subject_type, subject_id, item)

    def search(self, subject_type: str, subject_id: str) -> list[Evidence]:
        return rank_evidence(self._evidence.get((subject_type, subject_id), []))

    def extract(self, subject_type: str, subject_id: str, evidence: list[Evidence]) -> Extraction | None:
        item = self._extractions.get((subject_type, subject_id))
        if item is None or not evidence_supports_extraction(item, evidence):
            return None
        return item
