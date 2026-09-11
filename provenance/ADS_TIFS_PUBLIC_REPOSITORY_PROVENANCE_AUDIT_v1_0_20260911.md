# ADS-TIFS Public Repository Provenance Audit v1.0

**Date:** 2026-09-11  
**Status:** `FAIL_PUBLIC_REPOSITORY_CONFORMANCE / CORE_CANONICAL_HASHES_REMAIN_VALID`  
**New PGD:** `NO`  
**Scientific estimand change:** `NO`

## Trigger

The current manuscript/supplement names the public repository
`djokobandjur/vit-positional-adversarial` and cites Git commit
`952ff4e7b81a220c40bc63483d332dc4d25277a2`.

The intended ADS/TIFS repository is instead:

`djokobandjur/ads-vit-forensics`.

A direct repository audit was therefore opened before submission freeze.

## Finding 1 — the cited repository/commit is wrong

The cited commit

`952ff4e7b81a220c40bc63483d332dc4d25277a2`

is an actual commit in `djokobandjur/vit-positional-adversarial`, a different related
project. It is **not** an ADS/TIFS `ads-vit-forensics` repository identity.

Therefore the manuscript/supplement statement that uses that repository/commit as
ADS/TIFS training-source provenance is **RETIRED**.

This is a provenance error, not evidence that the canonical ADS result artifacts were
computed from that GitHub snapshot.

## Finding 2 — the correct public repository currently exists, but is stale

The correct repository is:

`djokobandjur/ads-vit-forensics`

and its `main` branch HEAD at audit time is:

`cb0709a78c88602a5a8fed314508d369f9849fcd`.

However, the repository is aligned to an older TIFS submission state rather than the
current corrected canonical pipeline.

The current public tree does **not** contain the canonical corrected source filenames:

- `ads_canonical_operator_v2_1.py`;
- `ads_attack_engine_v2_1.py`;
- `ads_primary_full_grid_v2_1.py`;
- `derive_damage_matched_v2_1.py`;
- `stage_b_saved_delta_reeval_v2_8_2.py`.

The public `scripts/ads_experiment.py` also exposes historical semantics that directly
conflict with the corrected manuscript/canonical operator:

1. It averages attention distributions across images before computing KL.
   The canonical ADS estimand instead computes per-image KL first, then averages images.

2. It adds `eps_kl = 1e-10` to probabilities and renormalizes them.
   The canonical operator uses native logits, float64 `log_softmax`, no additive
   probability floor, and no probability renormalization.

3. Its PE attack takes only `next(iter(ref_loader))`, i.e. one reference batch.
   The canonical attack objective uses the complete fixed 256-image reference cohort.

4. It switches the attacked model to `train()` during PGD.
   The canonical attack uses `eval()`.

5. It includes `inv_freq` in the RoPE attack selection.
   The canonical surface explicitly excludes `inv_freq`.

These are not cosmetic differences. The current public repository must therefore be
treated as **HISTORICAL / UNSYNCHRONIZED / NOT AUTHORITATIVE** for the corrected
manuscript.

## Finding 3 — the manuscript's canonical SHA-256 values were not computed from the wrong repository

The repository error does **not** invalidate the principal SHA-256 identities printed
in the manuscript and supplement.

Those hashes were independently bound to canonical execution packages, protocol locks,
data/cohort objects, result artifacts, and exact recovered source files.

In particular:

- ADS specification `093e...7362` remains valid.
- ADS implementation file `14d3...b4fd6` remains valid and matches the exact
  `ads_canonical_operator_v2_1.py` recovered from the canonical execution/source-audit
  package.
- FULL_GRID protocol lock `2b242...021f` remains valid.
- specificity protocol lock `c3fc...7565e` remains valid.
- reference/holdout/cache hashes remain valid.
- canonical full-grid/damage-matched/adaptive/Stage-B artifact hashes remain valid.

The only repository-specific identity in the reader-facing reproducibility statement
that fails is the 40-character Git commit/repository claim.

Ledger v1.19 itself remains byte-valid, but because it repeats the wrong repository
provenance it must no longer be the latest scientific authority.

## Submission decision

**DO NOT RESUBMIT the current v1.2 manuscript/supplement unchanged.**

Before final submission:

1. advance the scientific ledger and explicitly retire the wrong repository provenance;
2. correct the Reproducibility Statement so it describes the full
   manuscript → source → source SHA → protocol → oracle test → artifact → claim chain;
3. remove reader-facing terms such as `release-candidate bundle`;
4. synchronize `djokobandjur/ads-vit-forensics` to the corrected canonical source and
   reproducibility materials, or temporarily omit a public-repository claim;
5. only after synchronization, freeze and cite the **new exact commit SHA** from the
   correct repository;
6. rerun source-to-manuscript and repository-to-package conformance checks against that
   frozen commit.

Simply replacing `952ff...` by the current `cb0709...` is prohibited because the latter
still points to stale/superseded code.

## State classification

- `djokobandjur/vit-positional-adversarial @ 952ff...`:
  `RETIRED_WRONG_REPOSITORY_FOR_ADS_TIFS`.
- `djokobandjur/ads-vit-forensics @ cb0709...`:
  `HISTORICAL_UNSYNCHRONIZED_CURRENT_HEAD`.
- canonical execution/source artifacts in the project authority set:
  `VALID`.
- manuscript/supplement v1.2:
  `SUPERSEDED_PENDING_REPOSITORY_PROVENANCE_CORRECTION`.
