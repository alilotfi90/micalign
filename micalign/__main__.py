"""
Command-line entry point.

    python -m micalign --input  "sample images/samples" \
                       --calibration "sample images/caliberation" \
                       --output aligned_out
"""

import argparse
from .processor import AlignmentProcessor


def main():
    p = argparse.ArgumentParser(
        prog="micalign",
        description="Align MicaSense Altum-PT / RedEdge-P captures in a folder.")
    p.add_argument("--input", "-i", required=True,
                   help="folder of captures (IMG_xxxx_1..7.tif)")
    p.add_argument("--calibration", "-c", required=True,
                   help="reflectance-panel capture (folder, glob, or one band file)")
    p.add_argument("--output", "-o", required=True, help="output folder")
    p.add_argument("--reference-index", type=int, default=5,
                   help="band index to align to (default 5 = panchromatic)")
    p.add_argument("--no-ecc", action="store_true",
                   help="skip ECC; use the robust phase-correlation path only (faster)")
    p.add_argument("--interpolation", choices=["cubic", "linear", "lanczos"],
                   default="cubic", help="resampling kernel (default cubic)")
    p.add_argument("--no-crop", action="store_true",
                   help="do not crop output to the common valid region")
    p.add_argument("--save-tif", action="store_true",
                   help="also write the float32 reflectance multi-band TIFF "
                        "(off by default; only needed for a reflectance/NDVI pipeline)")
    p.add_argument("--no-rgb", action="store_true",
                   help="do not write the stretched RGB PNG")
    p.add_argument("--cir-preview", action="store_true",
                   help="also write a colour-infrared (NIR/Red/Green) PNG")
    p.add_argument("--no-qc", action="store_true",
                   help="do not write the per-capture alignment QC JSON")
    args = p.parse_args()

    AlignmentProcessor(
        input_dir=args.input,
        calibration=args.calibration,
        output_dir=args.output,
        reference_index=args.reference_index,
        use_ecc=not args.no_ecc,
        interpolation=args.interpolation,
        crop=not args.no_crop,
        save_rgb=not args.no_rgb,
        save_tif=args.save_tif,
        cir_preview=args.cir_preview,
        save_qc=not args.no_qc,
    ).run()


if __name__ == "__main__":
    main()
