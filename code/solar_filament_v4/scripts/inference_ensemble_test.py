from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import pandas as pd
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.common import encode_mask_to_rle, physical_image_key, IMAGE_EXTS
from src.infer_native import predict_native_image
from src.ensemble_merge import ensemble_instances
from src.postprocess import suppress_duplicates_and_make_exclusive


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--models", nargs=3, required=True)
    p.add_argument("--test-dir", required=True)
    p.add_argument("--tile-size", type=int, default=1024)
    p.add_argument("--stride", type=int, default=768)
    p.add_argument("--imgsz", type=int, default=1024)
    p.add_argument("--conf", type=float, default=0.30)
    p.add_argument("--merge-iou", type=float, default=0.10)
    p.add_argument("--merge-containment", type=float, default=0.45)
    p.add_argument("--max-instances", type=int, default=15)
    p.add_argument("--duplicate-iou", type=float, default=0.80)
    p.add_argument("--duplicate-containment", type=float, default=0.88)

    p.add_argument("--ensemble-match-iou", type=float, default=0.25)
    p.add_argument("--ensemble-match-containment", type=float, default=0.55)
    p.add_argument("--min-model-support", type=int, default=2)
    p.add_argument("--pixel-vote-threshold", type=float, default=0.50)

    p.add_argument("--device", default="0")
    p.add_argument("--output", default="submission_v4_ensemble.csv")
    return p.parse_args()


def unpack_prediction_output(prediction_output):
    """
    Compatible with:
      V3-style: return final, diagnostics
      V4-style: return final
    """
    if (
        isinstance(prediction_output, tuple)
        and len(prediction_output) == 2
        and isinstance(prediction_output[1], dict)
    ):
        return prediction_output[0]
    return prediction_output


def main():
    args = parse_args()
    models = [YOLO(p) for p in args.models]

    image_paths = sorted(
        p for p in Path(args.test_dir).iterdir()
        if p.is_file() and p.suffix.lower() in IMAGE_EXTS
    )

    rows = []

    for n, p in enumerate(image_paths, start=1):
        image = cv2.imread(str(p), cv2.IMREAD_COLOR)
        if image is None:
            raise RuntimeError(f"Could not read {p}")

        per_model = []

        for model in models:
            prediction_output = predict_native_image(
                model,
                image,
                tile_size=args.tile_size,
                stride=args.stride,
                imgsz=args.imgsz,
                conf=args.conf,
                merge_iou=args.merge_iou,
                merge_containment=args.merge_containment,
                max_instances=args.max_instances,
                duplicate_iou=args.duplicate_iou,
                duplicate_containment=args.duplicate_containment,
                device=args.device,
            )

            # FIX: strip diagnostics if predict_native_image returns
            # (final, diagnostics).
            pred = unpack_prediction_output(prediction_output)
            per_model.append(pred)

        fused = ensemble_instances(
            per_model,
            match_iou=args.ensemble_match_iou,
            match_containment=args.ensemble_match_containment,
            min_model_support=args.min_model_support,
            pixel_vote_threshold=args.pixel_vote_threshold,
        )

        final = suppress_duplicates_and_make_exclusive(
            [m for m, _, _ in fused],
            [score for _, score, _ in fused],
            max_instances=args.max_instances,
            duplicate_iou=args.duplicate_iou,
            duplicate_containment=args.duplicate_containment,
        )

        image_id = physical_image_key(p.name)

        for idx, (mask, score) in enumerate(final, start=1):
            rows.append(
                {
                    "filament_id": f"{image_id}_{idx}",
                    "segmentation_rle": encode_mask_to_rle(mask),
                }
            )

        print(
            f"[{n}/{len(image_paths)}] {image_id}: "
            f"model counts={[len(x) for x in per_model]} "
            f"fused={len(final)}"
        )

    df = pd.DataFrame(rows, columns=["filament_id", "segmentation_rle"])

    if not df["filament_id"].is_unique:
        raise RuntimeError("Duplicate filament_id values generated.")
    if df.isna().any().any():
        raise RuntimeError("Null value detected.")

    df.to_csv(args.output, index=False)

    print(f"\nSaved: {args.output}")
    print(f"Rows : {len(df)}")


if __name__ == "__main__":
    main()
