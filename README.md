# micalign

## Developers

- Ali Lotfi
- Motasin Akib
- Claude Opus 4.8 (Anthropic)

Robust band-to-reference alignment for **MicaSense Altum-PT / RedEdge-P** captures.

Point it at a folder of 7-band captures and a reflectance-panel calibration
capture; it writes a co-registered, radiometrically-calibrated **multi-band
reflectance TIFF** (plus an alignment QC JSON) for every capture in the folder.

This package does alignment only — no pansharpening.

---

## What it fixes

Two things that make MicaSense alignment outputs look soft or fringed:

1. **Repeated resampling.** The usual path resamples each band three times
   (lens undistortion → RigRelatives warp → ECC warp), all bilinear, which
   compounds blur. `micalign` composes lens undistortion + RigRelatives + the
   residual alignment and applies them in a **single `INTER_CUBIC` resample**.
   On Altum-PT samples this measured ~1.5–3× sharper than the three-stage path.

2. **Fragile / silent ECC.** OpenCV's ECC can fail to converge on repetitive
   crop texture and (in the original code) silently fall back to identity,
   leaving a band ~20 px misaligned — which shows up as colour fringing and,
   downstream, large NDVI error. `micalign` tries ECC but falls back to a
   **validated phase-correlation translation**, choosing whichever candidate
   scores best against the reference (gradient NCC). Every band records its
   method, score, ECC cc, phase response, residual shift, and a `degraded`
   flag — **failures are never silent.**

---

## Install

```bash
pip install -r requirements.txt
# (numpy opencv-python tifffile imageio scikit-image matplotlib
#  pyexiftool pytz pysolar pyzbar packaging)
```

System dependencies (the vendored MicaSense library needs these):

```bash
# Debian/Ubuntu
sudo apt-get install libimage-exiftool-perl libzbar0
# macOS
brew install exiftool zbar
```

`exiftool` reads the per-band calibration metadata (RigRelatives, radiometric
coefficients). `zbar` is needed for reflectance-panel QR detection.

The MicaSense `imageprocessing` library is vendored under
`micalign/vendor/micasense/`. If it is missing, fetch it with:

```bash
python -m micalign.bootstrap
```

---

## Usage

Expected folder layout (matches the sample data):

```
sample images 4/
  caliberation/      IMG_0000_1.tif ... IMG_0000_7.tif      <- panel capture
  sample images/     IMG_0010_1.tif ... IMG_0010_7.tif      <- one or more captures
```

### Command line

```bash
python -m micalign \
  --input       "sample images 4/sample images" \
  --calibration "sample images 4/caliberation" \
  --output      "aligned_out"
```

### Python

```python
from micalign import AlignmentProcessor

AlignmentProcessor(
    input_dir   = "sample images 4/sample images",
    calibration = "sample images 4/caliberation",
    output_dir  = "aligned_out",
).run()
```

All captures found in `input_dir` are processed with the shared panel
irradiance from the calibration capture.

---

## Output

By default, for each capture `IMG_xxxx` the package writes:

* **`IMG_xxxx_rgb.png`** — a stretched 8-bit natural-colour RGB image, ready to
  annotate and train YOLO on (LabelImg / CVAT / Roboflow / Ultralytics all read
  it directly). It uses the same global p2–p98 stretch as the original
  micasharpen RGB output.

* **`IMG_xxxx_align_qc.json`** — per-band `method` (`ecc` / `phase` / `factory`
  / `reference`), `alignment_score_ncc`, `ecc_cc`, `phase_response`,
  `residual_shift_px`, and `degraded`. Tiny; lets you skip any frame that didn't
  align cleanly before annotating. Disable with `--no-qc`.

Optional extra outputs:

* **`--save-tif`** → **`IMG_xxxx_aligned.tif`** — float32, multi-band,
  band-sequential (`planarconfig='separate'`, opens as one N-band raster in
  GDAL/QGIS), band order `[Blue, Green, Red, NIR, RedEdge, Panchro]` with the
  reference band last (NDVI: NIR = index 3, Red = index 2). This is the
  quantitative reflectance product for a reflectance/NDVI/SR pipeline — not
  needed for YOLO.
* **`--cir-preview`** → **`IMG_xxxx_cir.png`** — colour-infrared (NIR/Red/Green);
  vegetation is bright red, so band fringing and plot edges stand out.

By default the bands are aligned to the **panchromatic grid** (highest
resolution, best texture for robust alignment), so MS bands are upsampled ~2×.
This is the sharpest result without pansharpening — the package removes the
*added* blur (resampling + misalignment); the inherent MS-vs-pan resolution gap
is what a later pansharpening step would close. To align at native MS
resolution instead, pass `reference_index=` an MS band (e.g. `3` for NIR).

---

## Options

| Option | Default | Meaning |
|---|---|---|
| `reference_index` | `5` | band to align to (5 = panchromatic) |
| `use_ecc` | `True` | try ECC refinement; `False` = phase-only (faster) |
| `interpolation` | `"cubic"` | `cubic` (best for reflectance), `linear`, or `lanczos` |
| `crop` | `True` | crop output to the region where all bands overlap |
| `save_rgb` | `True` | write the stretched RGB PNG (the YOLO image) |
| `save_tif` | `False` | also write the float32 reflectance TIFF |
| `cir_preview` | `False` | also write the colour-infrared PNG |
| `save_qc` | `True` | write the per-capture alignment QC JSON |

CLI flags: `--reference-index`, `--no-ecc`, `--interpolation`, `--no-crop`,
`--save-tif`, `--no-rgb`, `--cir-preview`, `--no-qc`.
