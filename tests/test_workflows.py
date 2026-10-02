"""Behavior tests for the public algorithms and their audit boundaries."""

from __future__ import annotations

import tempfile
import unittest
from datetime import date
from pathlib import Path

from hedge_workflows.checkpoint import CheckpointStore, input_digest
from hedge_workflows.evidence import Evidence, Extraction, OfflineEvidenceAdapter, evidence_supports_extraction
from hedge_workflows.nav import assess_coverage, plan_requests, read_observations, read_tickers
from hedge_workflows.provenance import read_claims, validate_claims
from hedge_workflows.screening import Fund, fuse_signals


EXAMPLES = Path(__file__).resolve().parents[1] / "examples"


class NavTests(unittest.TestCase):
    def test_request_plan_is_contiguous_and_deduplicated(self) -> None:
        plan = plan_requests([" A Equity ", "A   Equity", "B Equity"], date(2025, 1, 1), date(2025, 3, 31), batch_size=1, window_months=1)
        self.assertEqual(len(plan), 6)
        self.assertEqual({item.tickers for item in plan}, {("A EQUITY",), ("B EQUITY",)})
        for batch in (1, 2):
            windows = [item for item in plan if item.batch_number == batch]
            self.assertEqual(windows[0].start, date(2025, 1, 1))
            self.assertEqual(windows[-1].end, date(2025, 3, 31))
            self.assertTrue(all(a.end.toordinal() + 1 == b.start.toordinal() for a, b in zip(windows, windows[1:])))

    def test_raw_coverage_keeps_missing_months_and_audits_matching(self) -> None:
        tickers = read_tickers(EXAMPLES / "sample_funds.csv")
        observations = read_observations(EXAMPLES / "sample_nav.csv")
        results, audit = assess_coverage(tickers, observations, date(2025, 1, 1), date(2025, 12, 31))
        statuses = {row.ticker: row.status for row in results}
        self.assertEqual(statuses["TEST-A EQUITY"], "Good")
        self.assertEqual(statuses["TEST-B EQUITY"], "Usable")
        self.assertEqual(statuses["TEST-C EQUITY"], "Sparse")
        self.assertEqual(statuses["TEST-D EQUITY"], "Stale")
        self.assertEqual(statuses["TEST-E EQUITY"], "Missing")
        self.assertEqual(audit["matched_fund_count"], 4)
        self.assertEqual(audit["unmatched_tickers"], ["TEST-X EQUITY"])
        self.assertEqual(len(audit["batch_audit"]), 2)
        self.assertEqual(next(row for row in results if row.ticker == "TEST-A EQUITY").unique_nav_quarters, 4)
        self.assertTrue(audit["no_forward_fill"])


class ScreeningTests(unittest.TestCase):
    def setUp(self) -> None:
        self.adapter = OfflineEvidenceAdapter(EXAMPLES / "sample_evidence.json")
        import csv

        with (EXAMPLES / "sample_funds.csv").open(newline="", encoding="utf-8") as handle:
            self.funds = {fund.fund_id: fund for fund in (Fund.from_row(row) for row in csv.DictReader(handle))}

    def screen(self, fund_id: str):
        fund = self.funds[fund_id]
        product_evidence = self.adapter.search("fund", fund.fund_id)
        manager_evidence = self.adapter.search("manager", fund.manager_id)
        return fuse_signals(
            fund,
            self.adapter.extract("fund", fund.fund_id, product_evidence),
            self.adapter.extract("manager", fund.manager_id, manager_evidence),
            product_evidence,
            manager_evidence,
        )

    def test_product_manager_conflict_requires_review(self) -> None:
        result = self.screen("F-B")
        self.assertEqual(result.product_exposure, "Ex-China")
        self.assertEqual(result.manager_background, "Mainland-linked")
        self.assertEqual(result.bucket, "Need Manual Check")
        self.assertEqual(result.manual_priority, "Medium")

    def test_regional_and_unknown_are_not_direct_exposure(self) -> None:
        self.assertEqual(self.screen("F-C").bucket, "Asia/HK Watchlist")
        self.assertEqual(self.screen("F-E").bucket, "Need Manual Check")
        self.assertEqual(self.screen("F-F").bucket, "Unknown Low Priority")
        self.assertEqual(self.screen("F-A").bucket, "Confirmed China-related")

    def test_uncited_or_snippet_only_high_claim_is_rejected(self) -> None:
        row = Evidence("fund", "F-Z", "https://example.org/snippet", "search_snippet", "China exposure", "2025-12-31")
        extraction = Extraction("fund", "F-Z", "China Direct", "High", row.source_url, "China exposure", "Search result")
        self.assertFalse(evidence_supports_extraction(extraction, [row]))
        other_subject = Evidence("manager", "M-Z", row.source_url, "official_site", "China exposure", "2025-12-31")
        self.assertFalse(evidence_supports_extraction(extraction, [other_subject]))


class AuditTests(unittest.TestCase):
    def test_checkpoint_rejects_changed_inputs(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            source = root / "input.csv"
            source.write_text("a\n", encoding="utf-8")
            checkpoint = root / "checkpoint.json"
            first = CheckpointStore(checkpoint, input_digest(source))
            first.save_result("A", {"bucket": "Unknown"})
            resumed = CheckpointStore(checkpoint, input_digest(source))
            self.assertTrue(resumed.contains("A"))
            source.write_text("b\n", encoding="utf-8")
            with self.assertRaises(ValueError):
                CheckpointStore(checkpoint, input_digest(source))

    def test_evidence_matrix_marks_conflicts_and_scope_changes(self) -> None:
        audit = validate_claims(read_claims(EXAMPLES / "sample_claims.jsonl"))
        types = {item["type"] for item in audit["issues"]}
        self.assertIn("conflicting_values", types)
        self.assertIn("incomparable_unit_or_denominator", types)
        self.assertEqual(audit["claim_count"], 3)


if __name__ == "__main__":
    unittest.main()
