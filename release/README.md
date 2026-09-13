# Pre-Zenodo release freeze

Status: **PRE_ZENODO_RELEASE_READY** (2026-09-13).

The Git repository is the lightweight code/source/provenance companion. The large archival payload is intentionally not committed to Git.

- audited canonical source/protocol snapshot cited by the manuscript: `ac580b9524c287882d97f4660fd7bf7791ea0c73`
- scientific authority: Ledger v1.31, SHA-256 `50efb67906fa9bc8058c3843cd2036dce25169e0fdf7720b190271881423a9b4`
- final submission wrapper SHA-256: `ef34d7e899bd0c1e05dd528a53de7f52af21779c8b68862ec505b91c3f374f0b`
- final compact replication package: `ADS_TIFS_SUBMISSION_REPLICATION_PACKAGE_v1_5_POST_v1_31_FREEZE_20260913.zip`, 368,622,268 bytes, SHA-256 `80f421a22c878a95809c62fea8ae6a1711cef9db4547bc10b1835650fed82943`
- Zenodo concept DOI: `10.5281/zenodo.19844729`
- version-specific DOI: **pending publication of the new Zenodo version**

The complete replication ZIP is the file to archive as the new Zenodo version. It already contains the frozen submission wrapper, current main/supplement, scientific-authority ledger, final audit/checklist, numerical dependencies, recovered EXP-019 canonical artifacts, figure sources, and reproduction scripts.

The 2,387 saved B1.5 attack deltas (~401 GB) remain a separately retained byte-level audit payload; they are not required for the statistical reproductions in the compact archive.

No scientific result changes are pending. After the Zenodo version DOI exists, only archival identifier/citation metadata should be updated, followed by a final identifier cross-check.
