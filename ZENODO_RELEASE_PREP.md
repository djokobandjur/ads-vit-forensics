# Zenodo synchronized-release preparation

Status: **DRAFT PREPARATION / DO NOT PUBLISH YET**

Proposed next software version: `v2.1.0`.

Persistent concept DOI:

`10.5281/zenodo.19844729`

The concept DOI represents all versions and resolves to the latest published Zenodo release.

## Audited public source identity

Repository:

`https://github.com/djokobandjur/ads-vit-forensics`

Immutable audited source/protocol snapshot:

`ac580b9524c287882d97f4660fd7bf7791ea0c73`

Scientific authority:

`ADS_TIFS_SCIENTIFIC_RESULTS_AND_INTERPRETATION_LEDGER_UPDATED_POST_GITHUB_CONFORMANCE_v1_21_20260911.md`

SHA-256:

`916865096723af369d51a81d648a3d19491b76bc598914509f5b3b99db008938`

## Prepared archival payloads

Compact audited GitHub source/protocol package:

`ADS_TIFS_GITHUB_AUDITED_SOURCE_PACKAGE_ac580b9_20260911.zip`

SHA-256:

`c548d02bd278898f43e5531621363eaa4212ee5429ac016f25ba94e82dd399c8`

Prepublication reproducibility package:

`ADS_TIFS_REPRODUCIBILITY_PACKAGE_v1_3_PREPUBLICATION_20260911.zip`

SHA-256:

`b187d8adf1a70fa79e037a3db84299847a42d7ed32e393d08dfdfaf5302a826e`

The reproducibility package reruns the independent synthetic/oracle suite at packaging time: **36/36 PASS**.

These files are prepared locally for the next Zenodo version and are not claimed to be published Zenodo files yet.

## New-version workflow

1. Open the latest published `ads-vit-forensics` Zenodo record and choose **New version**.
2. Reserve/get the new version-specific DOI in the draft before publication.
3. Do not publish the draft yet.
4. Insert that reserved DOI into the main manuscript and supplement together with:
   - public GitHub URL;
   - audited Git snapshot `ac580b...`;
   - concept DOI `10.5281/zenodo.19844729`.
5. Rebuild and cross-audit the manuscript/supplement.
6. Replace provisional archival files in the Zenodo draft with the DOI-complete final package and verify all hashes.
7. Publish only after the final manuscript + supplement + GitHub + Zenodo cross-audit passes.

## Proposed release title

`djokobandjur/ads-vit-forensics: v2.1.0 — Canonical source-audited TIFS reproducibility release`

## Scientific scope

This is a provenance/reproducibility synchronization release. It introduces no new PGD, attack state, or scientific estimand. Historical pre-correction code is preserved as provenance and remains explicitly non-authoritative for corrected claims.

Known archival source gaps remain explicit: exact confirmatory adaptive v2.7.2 runner, exact specificity v2.6 runner, and original comparison LOCK-002 JSON bytes.
