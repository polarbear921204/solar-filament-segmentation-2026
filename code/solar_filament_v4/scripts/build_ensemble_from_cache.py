from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.common import decode_rle, encode_mask_to_rle
from src.ensemble_merge import cluster_instances_across_models, cluster_to_mask
from src.postprocess import suppress_duplicates_and_make_exclusive


def parse_args():
    p = argparse.ArgumentParser(description="Fuse cached V4 predictions offline and report 1/2/3-model support.")
    p.add_argument("--cache", default="v4_1_predictions.jsonl")
    p.add_argument("--output", default="submission_v4_1_baseline.csv")
    p.add_argument("--ensemble-match-iou", type=float, default=0.25)
    p.add_argument("--ensemble-match-containment", type=float, default=0.55)
    p.add_argument("--min-intersection", type=int, default=32)
    p.add_argument("--min-model-support", type=int, choices=[1, 2, 3], default=2)
    p.add_argument("--pixel-vote-threshold", type=float, default=0.50)
    p.add_argument("--max-instances", type=int, default=15)
    p.add_argument("--duplicate-iou", type=float, default=0.80)
    p.add_argument("--duplicate-containment", type=float, default=0.88)
    p.add_argument("--compare-to", default=None, help="Optional previous V4 CSV to check baseline reproduction.")
    return p.parse_args()


def main():
    args = parse_args()
    cache_path = Path(args.cache)
    if not cache_path.is_file():
        raise FileNotFoundError(cache_path)

    rows = []
    seen = set()
    support_before = Counter()
    support_kept_pre_nms = Counter()
    counts_by_model = [0, 0, 0]
    total_images = 0

    with cache_path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            obj = json.loads(line)
            image_id = str(obj["image_id"])
            if image_id in seen:
                raise RuntimeError(f"Duplicate image in cache: {image_id}")
            seen.add(image_id)
            total_images += 1
            h, w = int(obj["height"]), int(obj["width"])
            if len(obj["predictions"]) != 3:
                raise RuntimeError(f"Expected three model predictions for {image_id}")

            predictions_by_model = []
            for i, model_items in enumerate(obj["predictions"]):
                preds = []
                for item in model_items:
                    preds.append((decode_rle(item["rle"], h, w), float(item["score"])))
                predictions_by_model.append(preds)
                counts_by_model[i] += len(preds)

            clusters = cluster_instances_across_models(
                predictions_by_model,
                match_iou=args.ensemble_match_iou,
                match_containment=args.ensemble_match_containment,
                min_intersection=args.min_intersection,
            )

            fused = []
            for cluster in clusters:
                support = len({midx for midx, _, _ in cluster.members})
                support_before[support] += 1
                if support < args.min_model_support:
                    continue
                item = cluster_to_mask(
                    cluster,
                    min_model_support=args.min_model_support,
                    pixel_vote_threshold=args.pixel_vote_threshold,
                )
                if item is not None:
                    fused.append(item)
                    support_kept_pre_nms[support] += 1

            final = suppress_duplicates_and_make_exclusive(
                [m for m, _, _ in fused],
                [score for _, score, _ in fused],
                max_instances=args.max_instances,
                duplicate_iou=args.duplicate_iou,
                duplicate_containment=args.duplicate_containment,
            )
            for idx, (mask, _) in enumerate(final, 1):
                rows.append({
                    "filament_id": f"{image_id}_{idx}",
                    "segmentation_rle": encode_mask_to_rle(mask),
                })

            if total_images % 20 == 0:
                print(f"[{total_images}] cached images fused", flush=True)

    df = pd.DataFrame(rows, columns=["filament_id", "segmentation_rle"])
    if not df["filament_id"].is_unique:
        raise RuntimeError("Duplicate filament_id generated.")
    if df.isna().any().any():
        raise RuntimeError("Null RLE/ID generated.")
    df.to_csv(args.output, index=False)

    print("\n=== V4.1 CACHED FUSION DIAGNOSTICS ===")
    print(f"images processed        : {total_images}")
    print(f"raw model counts        : {counts_by_model}")
    print(f"clusters support=1      : {support_before[1]}")
    print(f"clusters support=2      : {support_before[2]}")
    print(f"clusters support=3      : {support_before[3]}")
    print(f"fused before NMS        : {sum(support_kept_pre_nms.values())}")
    print(f"final submission rows   : {len(df)}")
    print(f"final predicted images  : {df['filament_id'].str.rsplit('_', n=1).str[0].nunique() if len(df) else 0}")
    print(f"CSV saved               : {args.output}")

    if args.compare_to:
        old = pd.read_csv(args.compare_to)
        old_sorted = old.sort_values("filament_id").reset_index(drop=True)
        new_sorted = df.sort_values("filament_id").reset_index(drop=True)
        if old_sorted.equals(new_sorted):
            print("BASELINE REPRODUCED EXACTLY: IDs and RLE match the previous submission.")
        else:
            old_ids = set(old["filament_id"].astype(str))
            new_ids = set(df["filament_id"].astype(str))
            common = old_ids & new_ids
            om = old.set_index("filament_id")["segmentation_rle"]
            nm = df.set_index("filament_id")["segmentation_rle"]
            different = sum(om[i] != nm[i] for i in common)
            print("WARNING: BASELINE DOES NOT MATCH EXACTLY.")
            print(f"old rows={len(old)} new rows={len(df)}; added IDs={len(new_ids-old_ids)} missing IDs={len(old_ids-new_ids)} different RLE={different}")
            print("Before interpreting experimental differences, verify that cache generation used exactly the same models and inference source/version as your original V4 submission.")


if __name__ == "__main__":
    main()
