from __future__ import annotations

import argparse
import json
import shutil
import sys
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.common import annotation_to_mask, build_image_index, mask_to_single_polygon, resolve_image_path, stable_id
from src.tiling import make_tiles


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--manifest', default='dataset_yolo_v4_3fold_manifest.csv')
    p.add_argument('--fold', type=int, required=True, choices=[0, 1, 2])
    p.add_argument('--output-root', default='dataset_v4_folds')
    p.add_argument('--tile-size', type=int, default=1024)
    p.add_argument('--stride', type=int, default=768)
    p.add_argument('--min-visible-area', type=int, default=16)
    p.add_argument('--min-visible-ratio', type=float, default=0.02)
    return p.parse_args()


def load_coco(json_path, cache):
    if json_path in cache:
        return cache[json_path]
    with open(json_path, 'r', encoding='utf-8') as f:
        coco = json.load(f)
    images = {stable_id(x['id']): x for x in coco.get('images', [])}
    anns = defaultdict(list)
    for ann in coco.get('annotations', []):
        anns[stable_id(ann['image_id'])].append(ann)
    cache[json_path] = (images, anns)
    return cache[json_path]


def main():
    args = parse_args()
    df = pd.read_csv(args.manifest)
    out = Path(args.output_root) / f'fold{args.fold}'
    if out.exists():
        shutil.rmtree(out)
    for split in ('train', 'val'):
        (out / 'images' / split).mkdir(parents=True, exist_ok=True)
        (out / 'labels' / split).mkdir(parents=True, exist_ok=True)

    coco_cache = {}
    image_index_cache = {}
    patch_rows = []
    total_instances = 0

    for n, (_, row) in enumerate(df.iterrows(), start=1):
        split = 'val' if int(row['fold']) == args.fold else 'train'
        json_path = str(row['source_json'])
        images_dir = Path(str(row['source_images_dir']))
        image_id_key = str(row['source_image_id_json'])

        if str(images_dir) not in image_index_cache:
            image_index_cache[str(images_dir)] = build_image_index(images_dir)

        images, anns_by_image = load_coco(json_path, coco_cache)
        info = images[image_id_key]
        src = resolve_image_path(info.get('file_name', row['source_file_name']), image_index_cache[str(images_dir)])
        image = cv2.imread(str(src), cv2.IMREAD_COLOR)
        if image is None:
            raise RuntimeError(f'Could not read {src}')

        h, w = image.shape[:2]
        gt_masks = [annotation_to_mask(ann, h, w) for ann in anns_by_image.get(image_id_key, [])]

        for tile in make_tiles(w, h, args.tile_size, args.stride):
            patch = image[tile.y0:tile.y1, tile.x0:tile.x1]
            ph, pw = patch.shape[:2]
            stem = f"{row['sample_stem']}__y{tile.y0}_x{tile.x0}"
            img_out = out / 'images' / split / f'{stem}.png'
            lbl_out = out / 'labels' / split / f'{stem}.txt'
            cv2.imwrite(str(img_out), patch)

            lines = []
            visible_instances = 0
            for gt in gt_masks:
                full_area = int(gt.sum())
                if full_area == 0:
                    continue
                local = gt[tile.y0:tile.y1, tile.x0:tile.x1]
                visible_area = int(local.sum())
                if visible_area < args.min_visible_area:
                    continue
                if visible_area / full_area < args.min_visible_ratio:
                    continue
                poly = mask_to_single_polygon(local)
                if poly is None or len(poly) < 6:
                    continue

                arr = np.asarray(poly, dtype=np.float32).reshape(-1, 2)
                arr[:, 0] = np.clip(arr[:, 0] / pw, 0.0, 1.0)
                arr[:, 1] = np.clip(arr[:, 1] / ph, 0.0, 1.0)
                lines.append('0 ' + ' '.join(f'{v:.8f}' for v in arr.reshape(-1)))
                visible_instances += 1

            lbl_out.write_text(('\n'.join(lines) + '\n') if lines else '', encoding='utf-8')
            patch_rows.append({
                'patch_stem': stem,
                'split': split,
                'physical_key': row['physical_key'],
                'fold': int(row['fold']),
                'generated_image': str(img_out),
                'generated_label': str(lbl_out),
                'num_instances': visible_instances,
            })
            total_instances += visible_instances

        if n % 100 == 0:
            print(f'[{n}/{len(df)}] source records processed')

    pd.DataFrame(patch_rows).to_csv(out / 'patch_manifest.csv', index=False)
    yaml = f'''path: {out.resolve().as_posix()}\ntrain: images/train\nval: images/val\n\nnames:\n  0: filament\n'''
    (out / 'dataset.yaml').write_text(yaml, encoding='utf-8')

    train_patches = sum(r['split'] == 'train' for r in patch_rows)
    val_patches = sum(r['split'] == 'val' for r in patch_rows)
    print('\n=== FOLD PATCH SUMMARY ===')
    print('fold            :', args.fold)
    print('train patches   :', train_patches)
    print('val patches     :', val_patches)
    print('patch instances :', total_instances)
    print('dataset yaml    :', out / 'dataset.yaml')


if __name__ == '__main__':
    main()
