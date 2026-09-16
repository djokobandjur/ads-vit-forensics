# ADS-TIFS Submission Closure v1.5.2 — Final Audit

**Date:** 2026-09-16  
**Decision:** `PASS / SUBMISSION-READY`  
**Scientific change:** `NONE`  
**New experiment:** `NONE`  
**Ledger update:** `NONE`  
**Scientific authority:** Ledger v1.62  
**Ledger SHA-256:** `c3cc0a791517671e1e819c7bbd54518236d61a5a80bb4ec0a452b8f4ee4872d1`

## Scope

This closure resolves the final three editorial/provenance findings after the accepted six-page supplement candidate:

1. Training Table VII now spells out **`Label smoothing 0.1`** rather than the abbreviation `LS 0.1`.
2. Supplement Section S7 is titled **`Reproducibility Boundary and Archival Gaps`** and explicitly labels the unrecovered confirmatory-adaptive source archive, matched-specificity runner/notebook, and original comparison-protocol JSON bytes as **archival gaps**. The separately hash-bound 2,387 densified-stress deltas (~401 GB) remain disclosed.
3. Release-status wording is made truthful. No aligned post-audit public release is claimed. The main and supplement state that the currently public replication release predates this final audit closure and that current identities/traceability are bound by the supplement and submission manifest/Ledger v1.62.

Because item 3 exposed two stale claims in the closed v1.5.1 main, the main was reopened only for minimal provenance wording. No scientific statement, number, result, table, figure, or methodological interpretation was changed.

## Build gate

- Main PDF: **12 pages**
- Supplement PDF: **6 pages** (TIFS hard limit satisfied)
- Main overfull boxes: **0**
- Supplement overfull boxes: **0**
- Undefined references/citations: **0**
- Multiply-defined labels: **0**
- Bibliography: **35** `\\bibitem` entries
- Supplement WNIDs: **100 occurrences / 100 unique**
- PDF preflight: both PDFs openable, unencrypted, text PDFs

Underfull diagnostics are layout-only and were visually inspected; no clipping, overlap, or broken glyphs were observed.

## Render-diff gate

Against the exact v1.5.1 predecessor main PDF, only pages **4** and **12** change, corresponding to the operator-provenance sentence and Code/Data release-status wording. The other 10 pages are pixel-identical at the comparison render resolution.

Against six-page supplement candidate v3, only pages **2** and **6** change: page 2 spells out `Label smoothing`; page 6 updates the release boundary and explicit archival-gap terminology. Pages 1, 3, 4, and 5 are pixel-identical at the comparison render resolution.

All main figures are byte-identical to v1.5.1. All supplement figures and figure-source CSVs are byte-identical to candidate v3; no figure data were regenerated in this closure.

## Numerical-integrity gate

No scientific numerical value was edited. PDF numeric-token multisets for main are identical to v1.5.1. The supplement adds only provenance metadata text (`Ledger v1.62`) while preserving scientific/table values; `Label smoothing 0.1` replaces `LS 0.1` with the same value.

The previously removed public-model horizon table remains removed; no ledger expansion was performed to restore unsupported step-50/step-100 manuscript cells.

## Public-release boundary

The package deliberately does **not** claim that the pre-existing public GitHub/Zenodo replication release is aligned to all final audit/diagnostic branches. No public release was modified or published during this closure. The upload-ready manuscript is self-consistent with that state, and the package retains Ledger v1.62 plus SHA manifests as the current submission provenance layer.

## Current identities

- main TEX SHA-256: `edbbc881612626bb36e575aa7a68ac43414fcd8614e0a73c4152ed0341ec097f`
- main PDF SHA-256: `03fc7f138bd6f95e2ad31a47564bb8b9e955fa7eda65ecdb589f0a93673e5ea2`
- supplement TEX SHA-256: `5b18a764311fadd06bc70126eae28e10ec03f6411693dbcc0dd6f79713c4c12b`
- supplement PDF SHA-256: `3ecbc00bbdb2a57d9f682f6d7a01c43effaae782374a937943aab360df728823`
- bibliography SHA-256: `9dc7db45ab2b66b08c6e6c0e1ed5a63138c56c1532522fd0c743b7909dc4f6ef`

**Final decision:** `PASS / SUBMISSION-READY`. The six-page supplement and 12-page main are closed together as v1.5.2 submission provenance closure. No scientific ledger update is warranted.
