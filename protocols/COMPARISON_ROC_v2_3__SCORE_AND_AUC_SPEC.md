# Score and AUC specification — LOCKED

## Common pairing rule
Every detector score is a **same-image displacement score**.

Attack positive for image x_i:

```text
clean model on x_i  versus  exact canonical attacked model on x_i
```

Benign negative for transform T:

```text
clean model on x_i  versus  clean model on T(x_i)
```

Cross-image clean-half comparisons are not permitted in the primary corrected comparison/ROC estimand.

## ADS(L4)
Per-image canonical ADS at Layer 4: mean over heads/query rows of KL(clean row || test row), from native pre-softmax logits using float64 `log_softmax`, no additive floor and no renormalization. Operator hash is the global canonical ADS hash.

## Corrected Attn-L2
From Layer-4 native logits form float64 probabilities `p=exp(log_softmax(logits.float64,-1))`. For each image:

```text
f(x) = flatten(mean_query p_L4(x))   # [H,N] -> [H*N], H=12, N=197, D=2364
score = ||f_clean(x) - f_test(x)||_2
```

CLS key is retained. The historical `mean over key` feature is forbidden except as an optional negative control.

## Diagonal Mahalanobis
Use the same corrected key-salience feature. Across the 256 clean reference images compute per-coordinate sample variance (`ddof=1`) `v_j` and

```text
lambda = 1e-6 * mean(v_j) + 1e-18
score(x) = sqrt(sum_j( delta_f_j(x)^2 / (v_j + lambda) ))
```

This is a diagonal-variance-normalized displacement and must not be labeled full-covariance Mahalanobis.

## LogitKL
For each image, compare class-output distributions from clean and test conditions using float64 log-softmax:

```text
KL(softmax(z_clean(x)) || softmax(z_test(x)))
```

No additive smoothing/floor and no mean-distribution-across-images construction.

## Primary AUC
For each PE x seed x epsilon x method:

- positives: 256 attacked same-image scores;
- negatives: 256 identity scores + 9*256 transformed benign scores = 2560;
- AUC: standard rank AUC with 0.5 tie handling;
- compute within seed; then report mean +/- sample SD over six seeds; never pool seeds before AUC.

## Secondary diagnostics
1. `benign_only_auc`: same pooled rank AUC excluding identity zeros (2304 negatives).
2. `paired_benign_concordance`: for each image compare its attack score only against its nine benign scores, then average 256 image-level concordances.
3. `worst_benign_exceedance_rate`: fraction of images where attack score exceeds the maximum of that image's nine benign scores.

## Operational boundary rule
If a high-confidence boundary is reported, it is the smallest tested epsilon where **all six seed-level primary AUC values are >=0.99**. If this never occurs, write `not established within the locked grid`. Do not relax the criterion post hoc.

## Forbidden postprocessing
- no threshold-grid trapezoidal AUC as the primary AUC;
- no cross-image pseudo-ADS clean negatives;
- no interpolation of detector scores/AUC to damage-matched epsilon points;
- no sign reversal of score orientation after looking at results;
- larger score = more anomalous for all four methods, fixed a priori.
