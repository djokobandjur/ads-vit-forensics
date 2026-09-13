# Zenodo handoff — user action required

GitHub/source work is prepared through the pre-Zenodo freeze. Do **not** publish a Zenodo version until the remaining metadata choices below are confirmed.

## Archival payload

Primary file for the new Zenodo version:

`ADS_TIFS_SUBMISSION_REPLICATION_PACKAGE_v1_5_POST_v1_31_FREEZE_20260913.zip`

- bytes: `368622268`
- SHA-256: `80f421a22c878a95809c62fea8ae6a1711cef9db4547bc10b1835650fed82943`

The package already contains the frozen submission wrapper, scientific authority, final audit/checklist, numerical dependencies, recovered EXP-019 canonical artifacts, figure sources, and reproduction scripts.

## Stable identifiers already known

- repository: `https://github.com/djokobandjur/ads-vit-forensics`
- audited source/protocol snapshot cited by the manuscript: `ac580b9524c287882d97f4660fd7bf7791ea0c73`
- Zenodo concept DOI: `10.5281/zenodo.19844729`
- version-specific DOI: **pending the new version**

## Suggested Zenodo metadata

- Resource type: Software (archival reproducibility release)
- Title: `Attention Divergence Score: A Forensic Metric for Characterizing Parameter-Level Attacks in Vision Transformers`
- Creators, in order:
  1. Djoko Bandjur — Faculty of Technical Sciences, University of Pristina -- Kosovska Mitrovica
  2. Milos Bandjur — Faculty of Technical Sciences, University of Pristina -- Kosovska Mitrovica
- Visibility: Public
- Language: English
- Keywords: Vision Transformer; attention divergence; digital forensics; parameter tampering; positional encoding; adversarial attacks

## User confirmation required before publication

1. **Release/version label** for the new Zenodo version / GitHub release tag.
2. **License metadata.** The historical repository changelog states MIT for code and CC BY 4.0 for data/documentation, but the current repository does not carry canonical LICENSE files. Zenodo requires licensing terms, so the intended declaration(s) must be confirmed before publication.
3. Optional ORCID identifiers for either creator, if desired.
4. Whether to upload only the compact replication ZIP or also the small submission wrapper as a separate convenience file. The replication ZIP already contains the wrapper, so the single-file deposit is sufficient for completeness.

`CITATION.cff` is committed for GitHub citation support. A `.zenodo.json` file is intentionally **not** committed yet because Zenodo would prefer it over `CITATION.cff`, and the license/version fields should not be guessed.

After the version-specific DOI is assigned, update only archival identifier/citation metadata in the repository/manuscript and run a final DOI/Git/hash cross-check. No scientific rerun is expected.
