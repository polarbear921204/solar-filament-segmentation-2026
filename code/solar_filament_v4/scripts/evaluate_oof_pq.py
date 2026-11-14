from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import cv2
import pandas as pd
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.common import annotation_to_mask, build_image_index, resolve_image_path, stable_id
from src.infer_native import predict_native_image
from src.pq_metric import PQStats, panoptic_quality


def parse_args():
    p = argparse.ArgumentParser(
        description=(
            "Honest 3-fold OOF PQ. "
            "Each record is evaluated only by the model whose validation fold "
            "contains that record."
        )
    )
    p.add_argument(
        "--models",
        nargs=3,
        required=True,
        metavar=("FOLD0_PT", "FOLD1_PT", "FOLD2_PT"),
    )
    p.add_argument(
        "--manifest",
        default="dataset_yolo_v4_3fold_manifest.csv",
    )
    p.add_argument("--tile-size", type=int, default=1024)
    p.add_argument("--stride", type=int, default=768)
    p.add_argument("--imgsz", type=int, default=1024)
    p.add_argument("--conf", type=float, default=0.30)
    p.add_argument("--max-det-per-tile", type=int, default=30)

    p.add_argument("--merge-iou", type=float, default=0.10)
    p.add_argument("--merge-containment", type=float, default=0.45)
    p.add_argument("--merge-min-intersection", type=int, default=32)

    p.add_argument("--max-instances", type=int, default=15)
    p.add_argument("--min-area", type=int, default=0)

    p.add_argument("--duplicate-iou", type=float, default=0.80)
    p.add_argument("--duplicate-containment", type=float, default=0.88)

    p.add_argument("--device", default="0")
    p.add_argument("--output", default="eval_v4_oof.csv")
    return p.parse_args()


def load_coco(json_path: str, cache: dict):
    if json_path in cache:
        return cache[json_path]

    with open(json_path, "r", encoding="utf-8") as f:
        coco = json.load(f)

    images = {
        stable_id(img["id"]): img
        for img in coco.get("images", [])
    }

    anns_by_image = defaultdict(list)
    for ann in coco.get("annotations", []):
        anns_by_image[stable_id(ann["image_id"])].append(ann)

    cache[json_path] = (images, anns_by_image)
    return cache[json_path]


def unpack_prediction_output(prediction_output):
    """
    Compatible with both possible predict_native_image() return formats:

    V3-style:
        return final, diagnostics

    V4-style:
        return final

    Returns:
        final: list of prediction items
    """
    if (
        isinstance(prediction_output, tuple)
        and len(prediction_output) == 2
        and isinstance(prediction_output[1], dict)
    ):
        return prediction_output[0]

    return prediction_output


def extract_pred_masks(final):
    """
    Robustly extract masks from prediction items.

    Expected common form:
        [(mask, score), ...]

    This also tolerates tuples/lists with extra metadata:
        [(mask, score, ...), ...]
    """
    pred_masks = []

    for item in final:
        if isinstance(item, (tuple, list)):
            if len(item) == 0:
                continue
            mask = item[0]
        else:
            # Fallback if a future version returns masks directly.
            mask = item

        pred_masks.append(mask)

    return pred_masks


def main():
    args = parse_args()

    df = pd.read_csv(args.manifest)

    if "fold" not in df.columns:
        raise RuntimeError(
            "Manifest has no 'fold' column. "
            "Run make_3fold_split.py first."
        )

    invalid_folds = sorted(
        set(df["fold"].astype(int).unique()) - {0, 1, 2}
    )
    if invalid_folds:
        raise RuntimeError(
            f"Invalid fold values found in manifest: {invalid_folds}"
        )

    models = [
        YOLO(args.models[0]),
        YOLO(args.models[1]),
        YOLO(args.models[2]),
    ]

    coco_cache = {}
    image_index_cache = {}

    total = PQStats()
    fold_totals = [
        PQStats(),
        PQStats(),
        PQStats(),
    ]

    rows = []

    for n, (_, row) in enumerate(df.iterrows(), start=1):
        fold = int(row["fold"])

        # Honest OOF:
        # fold0 records -> fold0 model
        # fold1 records -> fold1 model
        # fold2 records -> fold2 model
        #
        # Each fold-specific model was trained using the other two folds and
        # held out this fold for validation.
        model = models[fold]

        json_path = str(row["source_json"])
        images_dir = Path(str(row["source_images_dir"]))
        image_id_key = str(row["source_image_id_json"])

        images, anns_by_image = load_coco(
            json_path,
            coco_cache,
        )

        if image_id_key not in images:
            raise KeyError(
                f"Image id {image_id_key} not found in {json_path}"
            )

        img_info = images[image_id_key]

        images_dir_key = str(images_dir)
        if images_dir_key not in image_index_cache:
            image_index_cache[images_dir_key] = build_image_index(
                images_dir
            )

        src = resolve_image_path(
            img_info.get(
                "file_name",
                row.get("source_file_name", ""),
            ),
            image_index_cache[images_dir_key],
        )

        image = cv2.imread(
            str(src),
            cv2.IMREAD_COLOR,
        )
        if image is None:
            raise RuntimeError(
                f"Could not read image: {src}"
            )

        prediction_output = predict_native_image(
            model,
            image,
            tile_size=args.tile_size,
            stride=args.stride,
            imgsz=args.imgsz,
            conf=args.conf,
            max_det_per_tile=args.max_det_per_tile,
            merge_iou=args.merge_iou,
            merge_containment=args.merge_containment,
            merge_min_intersection=args.merge_min_intersection,
            max_instances=args.max_instances,
            min_area=args.min_area,
            duplicate_iou=args.duplicate_iou,
            duplicate_containment=args.duplicate_containment,
            device=args.device,
        )

        final = unpack_prediction_output(
            prediction_output
        )

        pred_masks = extract_pred_masks(
            final
        )

        h, w = image.shape[:2]

        gt_masks = [
            annotation_to_mask(
                ann,
                h,
                w,
            )
            for ann in anns_by_image.get(
                image_id_key,
                [],
            )
        ]

        stats = panoptic_quality(
            gt_masks,
            pred_masks,
            iou_threshold=0.5,
        )

        total = total + stats
        fold_totals[fold] = fold_totals[fold] + stats

        rows.append(
            {
                "fold": fold,
                "physical_key": row["physical_key"],
                "source_file_name": img_info.get(
                    "file_name",
                    row.get("source_file_name", ""),
                ),
                "gt_instances": len(gt_masks),
                "pred_instances": len(pred_masks),
                "tp": stats.tp,
                "fp": stats.fp,
                "fn": stats.fn,
                "sum_iou": stats.sum_iou,
                "record_pq": stats.pq,
            }
        )

        if n % 20 == 0 or n == len(df):
            print(
                f"[{n}/{len(df)}] "
                f"running OOF PQ={total.pq:.5f}"
            )

    pd.DataFrame(rows).to_csv(
        args.output,
        index=False,
    )

    print("\n=== V4 HONEST 3-FOLD OOF PQ ===")
    print(f"records       : {len(df)}")
    print(
        f"physical imgs : "
        f"{df['physical_key'].nunique()}"
    )

    for fold, s in enumerate(fold_totals):
        print(
            f"fold {fold}: "
            f"TP/FP/FN={s.tp}/{s.fp}/{s.fn} "
            f"SQ={s.sq:.5f} "
            f"RQ={s.rq:.5f} "
            f"PQ={s.pq:.5f}"
        )

    print(
        f"ALL TP/FP/FN : "
        f"{total.tp} / {total.fp} / {total.fn}"
    )
    print(f"ALL SQ        : {total.sq:.5f}")
    print(f"ALL RQ        : {total.rq:.5f}")
    print(f"ALL PQ        : {total.pq:.5f}")
    print(f"details       : {args.output}")


if __name__ == "__main__":
    main()
