# Final TIFS corrected attack-generator specification

Normative primary generator:

```text
CE-only projected ascent
full fixed 256-image ImageNet-100 reference mean CE
model.eval()
20 steps
alpha = 0.1 * epsilon
1 deterministic trajectory
zero delta initialization
raw final iterate after step 20
PE-only surface
per-buffer deltas
absolute per-group L_inf box
no attacked biases
```

The generator must apply `base + current_delta` before every gradient evaluation. Missing attacked gradients are fatal.

The completed v1.6→v1.6.4 optimizer study is retained as characterization evidence; it does not authorize silent best-so-far selection, horizon extension, annealing, or a claim of global optimality in the main canonical grid. Any such generator change requires a new protocol lock and re-localization of the operating grid.
