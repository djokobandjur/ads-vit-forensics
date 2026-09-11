# Stage-B saved-delta reevaluation — finalization recovery note

The exact executed Stage-B runner identity is:

`stage_b_saved_delta_reeval_v2_8_2.py`

SHA-256:

`d83219e566ab4829de57ac875d561b8e4f6b47aa19b61531a77ed5b4f00e1add`

The executed runner contains a known **post-computation metadata-finalization NameError**: a final metadata statement referenced `hold_stream`, while the measured canonical holdout variable was `hold_canonical_stream`.

This failure occurred **after all scientific cell computations and principal outputs had already been written**. The recovery record `FINALIZATION_RECOVERY_v2_8_2_1.json` established the completed scientific state and performed finalization only; it did **not** run new PGD, model inference, score computation, or AUC computation.

Recovered/completed evidence includes:

- 228/228 source saved-delta identities verified;
- 228 cell JSON records;
- 456 cell NPZ artifacts;
- 24 calibration NPZ artifacts;
- 912 seed-wise AUC rows;
- full-accuracy reconstruction maximum discrepancy approximately `1.4e-14` percentage points;
- identity-state score maximum exactly zero;
- failed runner SHA-256 `d83219e566ab4829de57ac875d561b8e4f6b47aa19b61531a77ed5b4f00e1add`;
- holdout canonical-identity stream SHA-256 `bd1acb20fbcf99b625e899885449eceaa2cdec353713e567892c4555a33e41a1`;
- holdout image stream SHA-256 `8dce76f0fb77163dbb9b020ff8c5b258b060dc074dd610f265939268fa757dc2`.

Canonical status of the result package is therefore `PASS_EXECUTION_RECOVERED_AFTER_FINALIZATION_NAMEERROR`.

## Reproducibility interpretation

The exact runner should be preserved unchanged as the executed historical source. It must **not** be silently edited and then represented as the executed byte identity. A separately corrected rerunnable finalizer may be provided if useful, but it must carry a different filename/hash and explicit derivative status.
