# Damage-matched analysis implementation specification

This file is normative for post-processing the locked fixed-absolute grid.

## Definitions

For each PE family and seed, let `D_i = clean_accuracy - attacked_accuracy(epsilon_i)` in percentage points and let `M_i` be any measured ADS quantity.

For target damage `d*`, locate the first adjacent epsilon pair `(i, i+1)` in ascending epsilon for which `D_i <= d* <= D_{i+1}` or `D_i >= d* >= D_{i+1}`. Exact hits use the exact cell.

If multiple crossings exist, choose the first crossing (smallest epsilon interval) and set `nonmonotonic_damage=true`.

For a non-exact crossing, define:

```text
w = (d* - D_i) / (D_{i+1} - D_i)
```

and interpolate:

```text
epsilon* = (1-w) epsilon_i + w epsilon_{i+1}
M*       = (1-w) M_i       + w M_{i+1}
```

Apply the identical `w` to each of the 12 layer values and every element of the `[256,12]` per-image ADS matrix.

No extrapolation. If `D_{i+1} == D_i` and the target is not exactly equal to that value, the pair cannot bracket the target.

## Primary target

`d* = 5.0 pp`. Must be estimable for all 24 PE×seed units. Otherwise stop and create a separately locked targeted-refinement package.

## Secondary targets

`10.0 pp` and `20.0 pp`. A target is reported cross-family only if all 24 units bracket it. Otherwise report NOT_ESTIMABLE; do not substitute another target.

## Severe-compromise threshold

For each PE×seed, target attacked accuracy is `0.5 * clean_accuracy`, equivalently damage `0.5 * clean_accuracy` pp. Use the same first-crossing/no-extrapolation rule. Severe compromise is a separate operating point, not the primary 5-pp matched-damage estimand.

## Output requirements

Save one row per PE×seed×target containing endpoint epsilons, endpoint damages, interpolation weight, interpolated epsilon, mean ADS, L4 ADS, all 12 layer ADS values, source cell JSON paths/SHA values, and an estimability flag. Save interpolated per-image arrays when both endpoint NPZs are present.
