# Final release-freeze status

Date: 2026-09-13  
Status: **PASS / PRE-ZENODO RELEASE READY**

This record summarizes the final joint manuscript/reproducibility freeze. It does not replace the full audit/checklist shipped in the replication archive.

## Frozen identities

- audited source/protocol snapshot: `ac580b9524c287882d97f4660fd7bf7791ea0c73`
- Scientific Results and Interpretation Ledger v1.31 SHA-256: `50efb67906fa9bc8058c3843cd2036dce25169e0fdf7720b190271881423a9b4`
- submission wrapper SHA-256: `ef34d7e899bd0c1e05dd528a53de7f52af21779c8b68862ec505b91c3f374f0b`
- compact replication package SHA-256: `80f421a22c878a95809c62fea8ae6a1711cef9db4547bc10b1835650fed82943`

## Final gates

- main fresh compile: 12 pages; no overfull or unresolved references/citations;
- supplement fresh compile: 7 pages; no overfull or unresolved references/citations;
- shipping-vs-fresh render comparison: 0 changed pages (12/12 main, 7/7 supplement);
- terminology/cross-reference audit: PASS;
- dense-grid reproduction: 96/96, manuscript minimum `1.840`, PASS;
- decisive adaptive bootstrap: 20,000/20,000, q2.5 `0.6601787590136425`, PASS;
- EXP-019 headline: 72/72 seed-level ratios > 1; paired family/control mean range `6.54717399809--15.88419867155`, PASS;
- replication-package integrity: 44/44, failures=0, PASS.

## Release boundary

The only intentional external-payload exception is the separately retained set of 2,387 B1.5 saved `.pt` deltas (~401 GB), needed only for byte-level saved-delta reevaluation and not for the archived-array statistical reproductions.

The next release-critical action is Zenodo version publication and assignment of the real version-specific DOI. No scientific branch is open.
