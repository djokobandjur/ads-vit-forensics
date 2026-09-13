# Reproducibility and provenance

The corrected ADS/TIFS workflow separates two verification layers.

1. **Source conformance:** manuscript equation/prose -> exact source function -> source SHA-256 -> locked protocol -> independent oracle/unit test.
2. **Numerical replication:** canonical stored artifact/per-image arrays -> derived analysis/table -> manuscript number/claim.

The immutable audited source/protocol snapshot is `ac580b9524c287882d97f4660fd7bf7791ea0c73`. Source conformance/oracle records already present in this repository report 36/36 passing independent checks for the audited core pipeline.

## Final scientific/release authority

- Scientific Results and Interpretation Ledger v1.31 SHA-256: `50efb67906fa9bc8058c3843cd2036dce25169e0fdf7720b190271881423a9b4`
- final compact replication package SHA-256: `80f421a22c878a95809c62fea8ae6a1711cef9db4547bc10b1835650fed82943`
- final submission wrapper SHA-256: `ef34d7e899bd0c1e05dd528a53de7f52af21779c8b68862ec505b91c3f374f0b`
- Zenodo concept DOI: `10.5281/zenodo.19844729`
- version-specific DOI: pending publication of the new version

The full ledger and large numerical arrays are archival payloads; Git carries their exact identities plus canonical source/protocol/provenance records.

## Final independent gates

The frozen replication package passes:

- dense-grid reproduction: 96/96 cells, maximum absolute discrepancy `1.88332283102e-08`, global minimum `R_upper = 1.839663980099` (manuscript `1.840`);
- decisive adaptive bootstrap: 20,000/20,000 estimable, q2.5 / median / q97.5 = `0.6601787590136425 / 0.8010375307621824 / 0.8133185510338008`;
- EXP-019 5-pp headline reproduction: 72/72 seed-level ratios > 1; minimum seed-level ratio `4.52077855655`; paired family/control mean range `6.54717399809--15.88419867155` (manuscript `6.55--15.88`).

The complete scripts and numerical dependencies for these statistical reruns are in the compact replication archive prepared for Zenodo.

## Recovered EXP-019 provenance

The original FMLE Specificity v2.6 result bundle was recovered. The exact canonical damage-matched JSON/CSV and cell-level CSV are co-released inside the frozen replication package. This closes the previously open Table-V artifact-availability gap without changing any scientific result.

## External payload boundary

The compact archival package does not duplicate the 2,387 saved B1.5 `.pt` attack deltas (~401 GB). Their exact identities are retained separately for byte-level saved-delta re-evaluation. They are not required for the archived-array statistical reproductions.

## Known archival source gaps

The following are still preserved as explicit source-archive gaps rather than silently reconstructed:

- exact confirmatory adaptive v2.7.2 runner/source archive;
- exact Specificity v2.6 execution runner/notebook (the original completed result bundle itself has been recovered);
- original comparison LOCK-002 JSON bytes.

These do not create a detected numerical contradiction in the final reported results.

## Zenodo handoff

GitHub is ready for archival handoff. The remaining external actions are to publish/archive the frozen release through Zenodo, obtain the real version-specific DOI, then update archival identifier metadata in the manuscript/repository and run one final identifier cross-check. No scientific rerun is planned.
