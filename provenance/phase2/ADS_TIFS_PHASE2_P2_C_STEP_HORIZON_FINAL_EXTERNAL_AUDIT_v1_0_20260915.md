# ADS–TIFS P2-C Post-Closure Step-Horizon Diagnostic — Final External Audit v1.0

**Date:** 2026-09-15  
**Submitted bundle:** `results_p2c_step_horizon_diagnostic_v1_0.zip`  
**Submitted-bundle SHA-256:** `b7b59d61d17aaa381f840559d1c0638c434ec0e6fb83276045389707b61f6bb1`  
**Protocol lock SHA-256:** `8f6b134a6abdcd9826c862ea262dae01d1881ad3eeab158c393662f5184053e9`  
**Frozen step-20 baseline-authority SHA-256:** `e33cb781a2905e6a3167435164bc803e1e2784d6391c5289ed0e190426cc1138`  
**Trajectory runner SHA-256:** `fe8eff94f8ff12d52a0e3a2cf993877eba91580032abd94ae07aad2da57e708d`  
**Merge runner SHA-256:** `7b08c337874c90c943cb7321e0b9bf23cf7c59aa9d595775aeb5ea973f57a255`  
**Merged result JSON SHA-256:** `7c73db822d03fed642f00abdcc1893c2d9afd28ee434a84a2212884d93f19e0e`  
**Merged summary SHA-256:** `f25852cc4e6150670b60b2a488f672d4ae7e3ea71e925da21c7a2e9cf6401ab6`  
**Audit decision:** `PASS — COMPLETE / PROVENANCE VERIFIED`  
**Scientific status:** `POST-CLOSURE DIAGNOSTIC COMPLETE`

## 1. Artifact integrity and expected counts

The submitted tree contains exactly:

- 8 completed trajectory cells;
- 32 saved snapshot delta artifacts (20/50/100/200 for each trajectory);
- 32 canonical per-image ADS artifacts;
- 2 model-level result JSON files;
- 1 merged diagnostic JSON and 1 merged summary.

No scientific cell is missing from the locked 8-trajectory design.

## 2. Step-20 reproduction gate

All eight trajectories pass the critical fail-closed reproduction gate.

For every model/epsilon cell:

- the saved step-20 delta is **bitwise identical** to the frozen historical P2-C delta;
- the 20 pre-update reference-CE values reproduce the historical trajectory within the locked `1e-8` tolerance;
- the 20 gradient-L-infinity values reproduce within the locked `1e-8` tolerance;
- the step-20 full-validation correct count is exact;
- the serialized `[256,12]` canonical per-image ADS matrix is **bitwise identical** to the historical P2-C artifact.

Therefore the longer runs are valid continuations of the exact historical 20-step trajectories, not a different attack implementation.

## 3. Delta and ADS gates

All 32 snapshot deltas are finite, within the exact L-infinity budget, and carry the locked protocol/model/epsilon/step metadata.

All 32 ADS NPZ files are finite `[256,12]` matrices and carry the canonical operator SHA:

`093e0e562f4e8ccc839a920adef251909a2f47f5368166b7d7b6ccf6411a7362`

Direct `saturated_fraction` and normalized RMS occupancy independently reproduce from the saved deltas. They remain distinct quantities.

## 4. DeiT result

| step | damage pp over locked epsilon window | range pp | roughness TV/|net| | mean saturated fraction | mean normalized RMS |
|---:|---|---:|---:|---:|---:|
| 20 | 2.978, 2.646, 17.178, 8.096 | 14.532 | 4.679 | 0.1747 | 0.6138 |
| 50 | 4.486, 3.548, 28.052, 11.948 | 24.504 | 5.568 | 0.2685 | 0.7262 |
| 100 | 6.302, 4.062, 22.666, 25.042 | 20.980 | 1.239 | 0.3178 | 0.7732 |
| 200 | 10.678, 4.866, 27.376, 28.308 | 23.442 | 1.659 | 0.3431 | 0.7980 |

The prespecified strong state+CE contraction gate is satisfied by **0/4** trajectories.

Per-epsilon state-change contraction ratios are:

`1.127, 1.106, 1.112, 1.106`

and CE-change contraction ratios are:

`0.809, 0.968, 0.751, 0.852`.

The mean absolute 20→200 damage change is **10.083 pp**.

**Locked classification:** `FIXED_STEP_TRAJECTORY_NOT_CONVERGED_BY_200`.

The key observation is that the delta-state movement does not contract from 50→100 to 100→200; all four state ratios are greater than 1. The fixed `alpha/epsilon=0.1` trajectory is therefore still materially moving at 200 steps.

## 5. RoPE result

| step | damage pp over locked epsilon window | range pp | roughness TV/|net| | mean saturated fraction | mean normalized RMS |
|---:|---|---:|---:|---:|---:|
| 20 | 4.634, 4.472, 8.416, 10.532 | 6.060 | 1.055 | 0.3925 | 0.7720 |
| 50 | 41.156, 49.466, 44.670, 50.360 | 9.204 | 2.042 | 0.3947 | 0.7804 |
| 100 | 46.234, 52.314, 46.818, 52.640 | 6.406 | 2.716 | 0.4444 | 0.8178 |
| 200 | 47.610, 53.796, 47.440, 54.188 | 6.748 | 2.933 | 0.4607 | 0.8287 |

The prespecified strong state+CE contraction gate is again satisfied by **0/4** trajectories.

Per-epsilon state-change contraction ratios are:

`0.899, 0.829, 0.929, 0.899`

and CE-change contraction ratios are:

`0.478, 0.383, 0.414, 0.471`.

The mean absolute 20→200 damage change is **43.745 pp**.

**Locked classification:** `FIXED_STEP_TRAJECTORY_NOT_CONVERGED_BY_200`.

RoPE therefore does not behave as the pre-execution directional prediction anticipated. The 20-step window was damage-smooth locally, but extending the same fixed-step trajectory produces much larger full-validation damage and the delta-state still fails the prespecified convergence criterion at 200 steps.

## 6. Interpretation

The diagnostic supports three narrow conclusions.

First, the public-model P2-C behavior is genuinely **attack-horizon sensitive**. Step 20 is not a stable approximation to the later fixed-step state: mean absolute 20→200 damage shifts are 10.083 pp for DeiT and 43.745 pp for RoPE.

Second, the stronger proposed explanation — that simply extending the same `alpha/epsilon=0.1` schedule to 200 steps would converge the iterate and regularize the epsilon-to-damage mapping — is **not supported**. Neither model satisfies the locked convergence gate in any of the four trajectories.

Third, this result does **not** establish an intrinsically non-smooth loss landscape. Because the step size is held fixed, the evidence is specifically for persistent fixed-step optimizer path dependence / nonconvergence through 200 steps. Distinguishing an oversized constant step from deeper multi-basin or non-smooth geometry would require a separately locked smaller-alpha or decayed-step experiment.

## 7. Relation to historical claims

This diagnostic does not retroactively invalidate:

- Phase 1;
- B1.5;
- the historical P2-C closure;
- the P2-D result.

P2-C remains `CLOSED_AFTER_LOCKED_REFINEMENT / QUALIFIED SUPPORT`.

The diagnostic refines the mechanistic interpretation of why direct damage matching was unstable in the public-model program: the sampled epsilon→damage mapping is entangled with a fixed-step optimizer trajectory that remains non-stationary even at 200 steps.

## 8. Final decision

`PASS_EXTERNAL_AUDIT / DIAGNOSTIC_COMPLETE`

Per-model locked state:

- DeiT: `FIXED_STEP_TRAJECTORY_NOT_CONVERGED_BY_200`
- RoPE: `FIXED_STEP_TRAJECTORY_NOT_CONVERGED_BY_200`

Cross-model state:

`NOT_TRIGGERED` for `SCHEDULE_ADEQUACY_IS_MODEL_DEPENDENT`.

No smaller-alpha, decayed-step, random-restart, or >200-step execution is authorized by this result.
