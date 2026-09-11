# Provenance and resume rules

1. First notebook cell: UID/cache bootstrap before any torch import.
2. Fail closed on reference-index SHA, transformed-reference SHA, operator hash, checkpoint SHA, protocol-lock SHA, PE topology, or script/config mismatch.
3. Cache transformed validation images once on GPU where memory permits; do not re-decode JPEGs inside PGD.
4. Save delta PT and canonical per-image ADS NPZ for every executed cell.
5. Delta PT metadata must be caller-supplied and carry the current protocol/execution identifiers.
6. Resume may skip only a fully verified cell. Never overwrite a verified cell automatically.
7. Preserve historical and corrected roots separately.
8. Every aggregate must record the exact source cell list and aggregation procedure.
