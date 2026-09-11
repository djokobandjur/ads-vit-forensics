# Benign transformation specification — LOCKED

The final comparison/ROC branch uses the fixed 256 ImageNet-100 reference images and creates transformed views of those **same images**.

## Identity
`identity`: unmodified reference image under the clean model. Its paired displacement score should be zero up to numerical tolerance. Identity is included as one equally weighted negative condition in the primary pooled AUC and is excluded in the benign-only sensitivity AUC.

## JPEG
Nominal conditions: `jpeg_q50`, `jpeg_q30`, `jpeg_q10`.

Procedure: inverse ImageNet normalization -> clamp to [0,1] -> RGB PIL image -> JPEG encode/decode at the requested quality -> float tensor [0,1] -> ImageNet normalization.

Executable package must explicitly fix JPEG encoder options rather than relying on hidden library defaults and must record Pillow/runtime version plus the transformed-tensor SHA for each condition.

## Gaussian blur
Nominal conditions: `blur_s1`, `blur_s2`, `blur_s3`.

Procedure: inverse ImageNet normalization -> clamp [0,1] -> RGB PIL -> `ImageFilter.GaussianBlur(radius={1,2,3})` -> float tensor [0,1] -> ImageNet normalization.

## Gaussian noise — CORRECTED
Nominal conditions: `noise_005`, `noise_010`, `noise_020`, interpreted as **pixel-space** standard deviation sigma={0.05,0.10,0.20} on [0,1] intensities.

Procedure: inverse ImageNet normalization -> add Gaussian N(0,sigma^2) -> clamp [0,1] -> ImageNet normalization.

Noise is fixed across PE families and model seeds. Per-image/per-level seed:

```text
payload = "ADS_TIFS_BENIGN_NOISE_v2_3|20260910|<reference_index>|<sigma:.2f>"
seed = int.from_bytes(SHA256(payload).digest()[:8], "little") % (2**63-1)
```

The executable package must generate the noise with the locked runtime, cache the resulting transformed tensors, and SHA-256 hash each transformed condition stream. A mismatch on resume is fail-closed.

## Explicit retirement
The historical behavior `normalized_tensor + randn_like * sigma` is retired and must not appear in final comparison/ROC results.
