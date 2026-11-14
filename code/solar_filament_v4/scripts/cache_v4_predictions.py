from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.common import IMAGE_EXTS, encode_mask_to_rle, physical_image_key
from src.infer_native import predict_native_image


def parse_args():
    p = argparse.ArgumentParser(description="Run three V4 YOLO models once and cache each model's instance masks as COCO RLE.")
    p.add_argument("--models", nargs=3, required=True, metavar=("FOLD0", "FOLD1", "FOLD2"))
    p.add_argument("--test-dir", required=True)
    p.add_argument("--output", default="v4_1_predictions.jsonl")
    p.add_argument("--resume", action="store_true", help="Resume an interrupted cache run. Config must match.")
    p.add_argument("--tile-size", type=int, default=1024)
    p.add_argument("--stride", type=int, default=768)
    p.add_argument("--imgsz", type=int, default=1024)
    p.add_argument("--conf", type=float, default=0.30)
    p.add_argument("--merge-iou", type=float, default=0.10)
    p.add_argument("--merge-containment", type=float, default=0.45)
    p.add_argument("--max-instances", type=int, default=15)
    p.add_argument("--duplicate-iou", type=float, default=0.80)
    p.add_argument("--duplicate-containment", type=float, default=0.88)
    p.add_argument("--device", default="0")
    p.add_argument("--expected-size", type=int, default=2048)
    return p.parse_args()


def unpack_prediction_output(output):
    if isinstance(output, tuple) and len(output) == 2 and isinstance(output[1], dict):
        return output[0]
    return output


def get_pairs(output):
    final = unpack_prediction_output(output)
    result = []
    for item in final:
        if not isinstance(item, (tuple, list)) or len(item) < 2:
            raise ValueError(f"Unexpected predict_native_image prediction item: {type(item)}")
        result.append((item[0], float(item[1])))
    return result


def main():
    args = parse_args()
    out = Path(args.output).resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    meta_path = Path(str(out) + ".meta.json")

    image_paths = sorted(p for p in Path(args.test_dir).iterdir()
                         if p.is_file() and p.suffix.lower() in IMAGE_EXTS)
    if not image_paths:
        raise RuntimeError("No test images found.")

    ids = [physical_image_key(p.name) for p in image_paths]
    if len(ids) != len(set(ids)):
        raise RuntimeError("Multiple test files resolve to the same image ID.")

    config = {
        "version": 1,
        "models": [str(Path(x).resolve()) for x in args.models],
        "test_dir": str(Path(args.test_dir).resolve()),
        "tile_size": args.tile_size, "stride": args.stride, "imgsz": args.imgsz,
        "conf": args.conf, "merge_iou": args.merge_iou,
        "merge_containment": args.merge_containment,
        "max_instances": args.max_instances, "duplicate_iou": args.duplicate_iou,
        "duplicate_containment": args.duplicate_containment,
        "expected_size": args.expected_size,
        "num_images": len(image_paths),
    }

    seen = set()
    if args.resume:
        if not out.exists() or not meta_path.exists():
            raise FileNotFoundError("--resume requires existing JSONL and matching metadata.")
        previous = json.loads(meta_path.read_text(encoding="utf-8"))
        if previous != config:
            raise RuntimeError("Cache configuration changed. Do not resume using different settings.")
        with out.open("r", encoding="utf-8") as f:
            for lineno, line in enumerate(f, 1):
                if not line.strip():
                    continue
                try:
                    obj = json.loads(line)
                    image_id = obj["image_id"]
                except Exception as e:
                    raise RuntimeError(f"Invalid JSONL at line {lineno}; repair/remove incomplete line before resume: {e}") from e
                if image_id in seen:
                    raise RuntimeError(f"Duplicate image in cache: {image_id}")
                seen.add(image_id)
        if not seen.issubset(set(ids)):
            raise RuntimeError("Cache contains IDs not present in the test folder.")
        print(f"Resuming: {len(seen)}/{len(image_paths)} images already cached.")
        mode = "a"
    else:
        if out.exists() or meta_path.exists():
            raise FileExistsError("Cache exists. Use --resume for matching settings or choose another --output.")
        meta_path.write_text(json.dumps(config, indent=2, ensure_ascii=False), encoding="utf-8")
        mode = "w"

    if len(seen) == len(image_paths):
        print("All images are already cached.")
        return

    models = [YOLO(model_path) for model_path in args.models]
    with out.open(mode, encoding="utf-8") as f:
        for index, path in enumerate(image_paths, 1):
            image_id = physical_image_key(path.name)
            if image_id in seen:
                continue
            image = cv2.imread(str(path), cv2.IMREAD_COLOR)
            if image is None:
                raise RuntimeError(f"Cannot read {path}")
            h, w = image.shape[:2]
            if (h, w) != (args.expected_size, args.expected_size):
                raise RuntimeError(f"Unexpected size for {path}: {w}x{h}")

            predictions_by_model = []
            for model in models:
                pred_output = predict_native_image(
                    model, image,
                    tile_size=args.tile_size, stride=args.stride, imgsz=args.imgsz,
                    conf=args.conf, merge_iou=args.merge_iou,
                    merge_containment=args.merge_containment,
                    max_instances=args.max_instances,
                    duplicate_iou=args.duplicate_iou,
                    duplicate_containment=args.duplicate_containment,
                    device=args.device,
                )
                pairs = get_pairs(pred_output)
                predictions_by_model.append([
                    {"score": score, "rle": encode_mask_to_rle(mask)}
                    for mask, score in pairs
                ])

            obj = {
                "image_id": image_id,
                "height": h, "width": w,
                "predictions": predictions_by_model,
            }
            f.write(json.dumps(obj, ensure_ascii=False, separators=(",", ":")) + "\n")
            f.flush()
            print(f"[{index}/{len(image_paths)}] {image_id} model counts={[len(x) for x in predictions_by_model]}", flush=True)

    print(f"\nCache saved: {out}")
    print("The cache can be reused for offline voting/matching experiments without running YOLO again.")


if __name__ == "__main__":
    main()
