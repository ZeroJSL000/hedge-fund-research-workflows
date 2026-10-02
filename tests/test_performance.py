"""Behavioral tests for comparable-sample and manager-ranking rules."""

from datetime import date
import unittest

from hedge_workflows.performance import (
    PerformancePolicy,
    assess_quality,
    build_comparable_sample,
    rank_managers,
    summarize_managers,
)


class PerformanceTests(unittest.TestCase):
    def setUp(self) -> None:
        self.policy = PerformancePolicy(
            as_of=date(2026, 9, 30),
            stale_after_days=300,
            numeric_bounds={"return_3y": (-0.9, 2.0), "sharpe_3y": (-5.0, 5.0)},
            minimum_manager_funds=2,
        )

    def row(self, fund: str, manager: str, return_3y: float) -> dict:
        return {
            "fund_id": fund,
            "manager_name": manager,
            "strategy": "Synthetic Strategy",
            "return_1y": 0.08,
            "return_3y": return_3y,
            "sharpe_3y": 0.8,
            "volatility_3y": 0.12,
            "max_drawdown": -0.16,
            "last_nav_date": "2026-08-31",
            "return_5y": 0.09,
            "aum_usd": 10_000_000,
        }

    def test_outlier_precedes_staleness_and_completeness(self) -> None:
        row = self.row("SYN-1", "Sample Manager", 2.5)
        row["last_nav_date"] = "2020-01-01"
        assessment = assess_quality(row, self.policy)
        self.assertEqual(assessment.tier, "F")
        self.assertIn("outside configured bounds", assessment.reasons[0])

    def test_stale_and_missing_risk_records_are_excluded_with_reasons(self) -> None:
        good = self.row("SYN-1", "Sample Manager", 0.12)
        stale = self.row("SYN-2", "Sample Manager", 0.10)
        stale["last_nav_date"] = "2020-01-01"
        incomplete = self.row("SYN-3", "Other Manager", 0.10)
        incomplete["sharpe_3y"] = None
        accepted, excluded = build_comparable_sample([good, stale, incomplete], self.policy)
        self.assertEqual([row["fund_id"] for row in accepted], ["SYN-1"])
        self.assertEqual([row["quality_tier"] for row in excluded], ["E", "C"])
        self.assertTrue(all(row["exclusion_reason"] for row in excluded))

    def test_median_summary_and_minimum_sample_gate(self) -> None:
        rows = [
            self.row("SYN-1", "Alpha Example", 0.10),
            self.row("SYN-2", "Alpha Example", 0.30),
            self.row("SYN-3", "Single Example", 0.90),
        ]
        comparable, excluded = build_comparable_sample(rows, self.policy)
        self.assertFalse(excluded)
        summary = summarize_managers(comparable, self.policy)
        ranked = rank_managers(summary, "median_return_3y")
        self.assertEqual(len(ranked), 1)
        self.assertEqual(ranked[0]["manager_name"], "Alpha Example")
        self.assertAlmostEqual(ranked[0]["median_return_3y"], 0.20)


if __name__ == "__main__":
    unittest.main()
