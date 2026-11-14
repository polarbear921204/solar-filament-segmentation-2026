#!/usr/bin/env python3
"""Apply the EXACT V11.2 trim1 mask rule to an already-reproduced V4 RLE CSV.

No YOLO inference or ground truth is used. This is a final-mask operation;
therefore it is a candidate experiment, not a validation of three-model PQ.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

IMAGE_EXTS = {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff'}
K3 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))


def physical_image_key(file_name: str) -> str:
    """Matches V4 src/common.py naming convention exactly."""
    stem = Path(str(file_name)).stem
    stem = re.sub(r'__.*$', '', stem)
    m = re.match(r'^\d{6}-(.+)$', stem)
    if m:
        stem = m.group(1)
    return stem


def contrast_map(image_bgr: np.ndarray) -> np.ndarray:
    """Byte-for-byte algorithm equivalent to V11.2 contrast_map."""
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
    fine = cv2.GaussianBlur(gray, (0, 0), 1.0)
    coarse = cv2.GaussianBlur(gray, (0, 0), 5.0)
    return coarse - fine


def refine_trim1(mask: np.ndarray, dark: np.ndarray) -> np.ndarray:
    """Exact V11.2 trim1 rule (only deletes pixels; never adds pixels)."""
    m = np.asarray(mask, dtype=bool)
    if not m.any():
        return m.copy()
    core = cv2.erode(m.astype(np.uint8), K3).astype(bool)
    rim = m & ~core
    bright_rim = rim & (dark < -1.5)
    if int(bright_rim.sum()) <= 0.15 * int(m.sum()):
        return m & ~bright_rim
    return m.copy()


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def get_ids(df: pd.DataFrame) -> dict[str, list[int]]:
    groups = defaultdict(list)
    for i, name in enumerate(df['filament_id']):
        name = str(name)
        sep = name.rfind('_')
        if sep <= 0 or not name[sep + 1:].isdigit():
            raise RuntimeError(f'Unexpected filament_id format: {name!r}')
        groups[name[:sep]].append(i)
    return dict(groups)


def run(args: argparse.Namespace) -> dict:
    try:
        from pycocotools import mask as mask_utils
    except ImportError as ex:
        raise RuntimeError('pycocotools is required: pip install pycocotools') from ex

    source = Path(args.submission).resolve()
    output = Path(args.output).resolve()
    if source == output:
        raise RuntimeError('Input and output paths must differ. Keep the original 0.38 CSV unchanged.')
    df = pd.read_csv(source, dtype=str, keep_default_na=False)
    expected = ['filament_id', 'segmentation_rle']
    if list(df.columns) != expected:
        raise RuntimeError(f'CSV columns must be {expected!r}, got {list(df.columns)!r}')
    if df.empty or (df[expected].eq('').any().any()) or not df['filament_id'].is_unique:
        raise RuntimeError('Input CSV is empty, has null/blank fields or duplicate filament_id.')
    groups = get_ids(df)
    image_paths = {}
    for p in Path(args.test_dir).iterdir():
        if p.is_file() and p.suffix.lower() in IMAGE_EXTS:
            key = physical_image_key(p.name)
            if key in image_paths:
                raise RuntimeError(f'Non-unique test image ID {key!r}: {image_paths[key]} and {p}')
            image_paths[key] = p
    if not image_paths:
        raise RuntimeError(f'No test images found under {args.test_dir}')
    unknown = sorted(set(groups) - set(image_paths))
    if unknown:
        raise RuntimeError(f'{len(unknown)} CSV image IDs have no test image, first: {unknown[:5]}')

    keys = list(groups)
    if args.limit_images:
        keys = keys[:args.limit_images]
        print('SMOKE TEST: this output is incomplete and MUST NOT be submitted to Kaggle.')
    if args.mode not in ('baseline', 'trim1'):
        raise RuntimeError('Unsupported mode')
    output_rles = df['segmentation_rle'].tolist()
    physical_stats = []
    for position, key in enumerate(keys, start=1):
        image = cv2.imread(str(image_paths[key]), cv2.IMREAD_COLOR)
        if image is None:
            raise RuntimeError(f'Cannot read {image_paths[key]}')
        h, w = image.shape[:2]
        if (h, w) != (2048, 2048):
            raise RuntimeError(f'{key}: expected 2048x2048, got {(h,w)}')
        dark = contrast_map(image) if args.mode == 'trim1' else None
        original_occupied = np.zeros((h, w), dtype=bool)
        refined_occupied = np.zeros((h, w), dtype=bool)
        num_changed = 0
        pixels_removed = 0
        total_before = 0
        for idx in groups[key]:
            rle = {'size': [h, w], 'counts': df.at[idx, 'segmentation_rle'].encode('utf-8')}
            try:
                original = mask_utils.decode(rle).astype(bool)
            except Exception as ex:
                raise RuntimeError(f'Failed decoding mask: {df.at[idx,"filament_id"]}') from ex
            if original.shape != (h, w) or not original.any():
                raise RuntimeError(f'Invalid or empty mask: {df.at[idx,"filament_id"]}')
            if np.any(original & original_occupied):
                raise RuntimeError(f'Original 0.38 CSV has overlapping masks at {key}')
            original_occupied |= original
            changed = refine_trim1(original, dark) if dark is not None else original
            # Pure trim must be a strict subset; preserve every ID and instance.
            if np.any(changed & ~original):
                raise RuntimeError(f'Unexpected added pixels: {df.at[idx,"filament_id"]}')
            if not changed.any():
                changed = original.copy()  # Same fallback as V11.2 refine_final()
            if np.any(changed & refined_occupied):
                raise RuntimeError(f'New overlapping mask: {df.at[idx,"filament_id"]}')
            refined_occupied |= changed
            removed = int(original.sum() - changed.sum())
            total_before += int(original.sum())
            pixels_removed += removed
            if removed:
                num_changed += 1
                encoded = mask_utils.encode(np.asfortranarray(changed.astype(np.uint8)))
                output_rles[idx] = encoded['counts'].decode('utf-8')
        physical_stats.append({
            'image_id': key, 'instances': len(groups[key]), 'masks_changed': num_changed,
            'pixels_removed': pixels_removed, 'pixels_before': total_before,
            'removed_fraction': pixels_removed / total_before if total_before else 0,
        })
        if position % 10 == 0 or position == len(keys):
            print(f'[{position}/{len(keys)}] {key} changed masks={num_changed} removed pixels={pixels_removed}', flush=True)
    # For a smoke run, do not inadvertently emit rows for the unprocessed images.
    indices = [i for key in keys for i in groups[key]] if args.limit_images else list(range(len(df)))
    output.parent.mkdir(parents=True, exist_ok=True)
    result = df.iloc[indices].copy()
    result['segmentation_rle'] = [output_rles[i] for i in indices]
    if not result['filament_id'].is_unique or result['segmentation_rle'].eq('').any():
        raise RuntimeError('Invalid output CSV')
    result.to_csv(output, index=False)
    audit = {
        'mode': args.mode,
        'smoke_incomplete_not_submittable': bool(args.limit_images),
        'input': str(source), 'output': str(output),
        'original_sha256': sha256(source), 'output_sha256': sha256(output),
        'images_with_predictions': len(keys), 'total_test_images': len(image_paths),
        'images_without_predictions_in_original': len(set(image_paths) - set(groups)),
        'mask_rows_input': len(df), 'mask_rows_output': len(result),
        'masks_changed': sum(x['masks_changed'] for x in physical_stats),
        'images_changed': sum(x['masks_changed'] > 0 for x in physical_stats),
        'pixels_removed': sum(x['pixels_removed'] for x in physical_stats),
        'pixels_before': sum(x['pixels_before'] for x in physical_stats),
        'number_of_output_overlaps': 0,
        'note': 'Fixed V11.2 trim1 rule applied after V4 three-model ensemble; no GT, no YOLO. OOF single-model gains do not guarantee test gains.',
    }
    summary_path = Path(args.summary)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(audit, ensure_ascii=False, indent=2), encoding='utf-8')
    stats_path = Path(args.per_image)
    stats_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(physical_stats).to_csv(stats_path, index=False)
    print('\n=== V11.3 FINAL MASK TRIM AUDIT ===')
    for k, v in audit.items():
        print(f'{k}: {v}')
    print(f'Saved: {output}\nAudit: {summary_path}\nPer-image: {stats_path}')
    return audit


def main():
    p = argparse.ArgumentParser(description='V11.3: apply frozen V11.2 trim1 to verified 0.38 V4 submission RLE masks')
    p.add_argument('--submission', required=True)
    p.add_argument('--test-dir', required=True)
    p.add_argument('--output', default='submission_v11_3_trim1.csv')
    p.add_argument('--summary', default='v11_3_trim1_audit.json')
    p.add_argument('--per-image', default='v11_3_trim1_images.csv')
    p.add_argument('--mode', choices=['baseline','trim1'], default='trim1')
    p.add_argument('--limit-images', type=int, default=0, help='Smoke run only; incomplete CSV cannot be submitted')
    a = p.parse_args()
    if a.limit_images < 0:
        p.error('--limit-images must be >=0')
    run(a)


if __name__ == '__main__':
    main()
