# Hedge Fund Research Workflows

This repository is a **public, synthetic-data reimplementation** of research workflow patterns. It is not a copy of the original internship code and cannot reproduce the internal research results. Every fund, manager, NAV observation, and numerical claim in `examples/` is invented for this repository.

## What the code demonstrates

| Workflow | Public algorithm | Boundary |
| --- | --- | --- |
| NAV coverage and comparable samples | Ticker normalization; deterministic ticker batches and date windows; raw NAV matching; monthly, quarterly, and trailing coverage; recency and gap diagnostics; `Missing`/`Stale`/`Sparse`/`Usable`/`Good` status; quality tiers and manager-level median summaries | No Bloomberg client, terminal access, original NAV series, fund universe, private cutoffs, or historical rankings |
| China relevance screening | Separate product-exposure and manager-ownership judgments; ranked search evidence; validated JSON extraction; conservative rule fusion; conflict and manual-review routing; input-bound checkpoints | No proprietary fund or manager names, search cache, commercial search/LLM calls, or actual candidate list |
| Research evidence matrix | Source URL and document title, page, observation date, unit, denominator, claim, qualifier, and numerical conflict checks | No third-party PDFs, internal report text, investment recommendations, or claim of factual verification |

The NAV and screening methods are inspired by data-quality and research-audit problems encountered during an internship. The implementation and example data here were written for public demonstration. Thresholds in the examples are illustrative and must be recalibrated for any real research question.

## Run the synthetic examples

Python 3.10 or later is required. The core package has no runtime dependencies.

```powershell
python -m pip install -e .
New-Item -ItemType Directory -Force .local-output | Out-Null

hedge-workflows nav --funds examples/sample_funds.csv --observations examples/sample_nav.csv --start 2025-01-01 --end 2025-12-31 --output .local-output/nav_coverage.csv

hedge-workflows screen --funds examples/sample_funds.csv --evidence examples/sample_evidence.json --checkpoint .local-output/screen_checkpoint.json --output .local-output/screening.jsonl

hedge-workflows performance --records examples/sample_performance.csv --policy examples/sample_performance_policy.json --output .local-output/performance.json

hedge-workflows provenance --input examples/sample_claims.jsonl --output .local-output/provenance_audit.json

python -m unittest discover -s tests -v
```

Generated results and checkpoints belong in `.local-output/`, which Git ignores. Real inputs should also stay outside the repository or under `.local-output/`. The `.gitignore` excludes CSV, JSON, JSONL, spreadsheets, PDFs, ZIPs, logs, and environment files by default; it allows only the six named synthetic example files back in.

## NAV method

`hedge_workflows.nav.plan_requests` creates a request plan from unique normalized tickers, inclusive date windows, and ticker batches. It does not call a market-data service. `assess_coverage` then compares raw observations with the fund universe and reports unmatched tickers, missing funds, batch counts, duplicate dates, non-positive values, the last observation, the longest gap, and coverage by calendar month and quarter. It never forward fills a missing NAV.

Status classification follows this order: no observations → `Missing`; excessive last-observation age → `Stale`; sufficiently fresh and continuous → `Good`; moderately recent and continuous → `Usable`; otherwise → `Sparse`. The policy is passed as a `CoveragePolicy`; CLI values are demonstration defaults. A pricing-frequency-aware production study would additionally calibrate expected observations for weekly, quarterly, or irregular disclosures.

`hedge_workflows.performance` builds an explicitly comparable subset before calculating manager summaries. It applies caller-supplied outlier bounds, staleness, and field-completeness checks, retains exclusion reasons and quality tiers, computes manager medians, and ranks only managers meeting the configured minimum fund count. These statistics describe the supplied sample, not investment merit.

The same functions can be used without the CLI:

```python
from datetime import date
from hedge_workflows.performance import PerformancePolicy, build_comparable_sample, summarize_managers

policy = PerformancePolicy(
    as_of=date(2025, 12, 31),
    stale_after_days=60,
    numeric_bounds={"return_3y": (-1.0, 5.0)},
    minimum_manager_funds=3,
)
accepted, excluded = build_comparable_sample(records, policy)
manager_summary = summarize_managers(accepted, policy)
```

Here, `records` is an iterable of dictionaries with fund-level performance fields. The values above are synthetic demonstration settings.

## Screening method

`classify_product` reads fund name, mandate, geography, and benchmark. `classify_manager` reads ownership notes and location. A Hong Kong address or an Asia mandate is a weak regional signal; a manager link never establishes direct product exposure. Explicit `ex-China` language is treated as an exclusion. `fuse_signals` sends contradictory signals to a manual queue.

`SearchProvider` and `JsonExtractor` are interfaces for external adapters. This repository ships `OfflineEvidenceAdapter` only. Its synthetic JSON file replays cited evidence and structured extractions without network access. A real adapter should return the same `Evidence` and `Extraction` records, and must protect credentials and comply with its source's access terms. An extraction is accepted only when its URL matches a retrieved record, its quote appears in that record, and the source quality supports the stated confidence. `extraction_prompt` provides a service-neutral JSON prompt schema. The CLI writes an input-bound checkpoint after each fund so an interrupted run can resume safely; changed inputs require a new checkpoint path.

No real search, LLM, or Bloomberg provider is included. The interfaces preserve the workflow structure while the runnable example stays fully offline.

The `review_top_rank` cutoff and all labels in this version are examples of the decision structure. They are not a disclosure of an employer's screening policy.

## Evidence matrix method

`Claim` records source provenance and the basis of a numerical statement. `validate_claims` flags missing provenance, duplicate identifiers, invalid dates, different values on a comparable basis, and unit or denominator mismatches. It does **structural validation only**: it does not inspect source documents or certify the truth of a claim. A researcher must still read the cited page and resolve any disagreement.

## Package map

```text
src/hedge_workflows/
  nav.py          Request planning and raw NAV diagnostics
  performance.py  Comparable sample and manager aggregation
  screening.py    Independent labels and conservative fusion
  evidence.py     Search/extraction interfaces and citation checks
  checkpoint.py   Atomic restart state and input hashes
  provenance.py   Evidence matrix validation
  cli.py          Reproducible local commands
examples/         Invented funds, observations, evidence, and claims
tests/            Behavior tests for key decision boundaries
```

No source dataset, third-party report, API credential, internal evidence cache, or employer-specific conclusion is included.
