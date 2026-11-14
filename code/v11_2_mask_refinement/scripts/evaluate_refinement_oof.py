#!/usr/bin/env python3
"""Single-inference, prediction-only mask refinement ablation on V4 fold-0 OOF.

Requires running from solar_filament project root with solar_filament_v4/src available.
No ground truth is used to choose or modify predictions. GT is used only for PQ.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path
import sys

import cv2
import numpy as np
import pandas as pd


def _load_v4_modules():
    cwd = Path.cwd()
    searches = [cwd, *cwd.parents, Path(__file__).resolve().parents[2]]
    for r in searches:
        if (r / 'solar_filament_v4' / 'src' / 'infer_native.py').exists():
            sys.path.insert(0, str(r / 'solar_filament_v4'))
            return
    raise RuntimeError('Cannot find solar_filament_v4/src; run from your solar_filament project root.')


_load_v4_modules()
from src.common import annotation_to_mask, build_image_index, resolve_image_path, stable_id
from src.infer_native import predict_native_image
from src.pq_metric import PQStats, panoptic_quality


VARIANTS = ('baseline', 'grow1', 'trim1', 'grow_trim1')
K3 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))


def contrast_map(image_bgr: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
    # Positive means locally darker than an approximately 11-pixel neighborhood.
    fine = cv2.GaussianBlur(gray, (0, 0), 1.0)
    coarse = cv2.GaussianBlur(gray, (0, 0), 5.0)
    return coarse - fine


def refine_one(mask: np.ndarray, dark: np.ndarray, variant: str) -> np.ndarray:
    m = np.asarray(mask, dtype=bool)
    if variant == 'baseline' or not m.any():
        return m.copy()
    m8 = m.astype(np.uint8)
    # Only adjust one-pixel rim: we cannot recover a completely absent instance.
    if variant in ('trim1', 'grow_trim1'):
        core = cv2.erode(m8, K3).astype(bool)
        rim = m & ~core
        bright_rim = rim & (dark < -1.5)
        # Safety guard: never remove >15% of a particular thin mask.
        if int(bright_rim.sum()) <= 0.15 * int(m.sum()):
            m = m & ~bright_rim
    if variant in ('grow1', 'grow_trim1'):
        expanded = cv2.dilate(m.astype(np.uint8), K3).astype(bool)
        # Require the new pixel to be locally dark. The threshold is absolute
        # on the input's 8-bit intensities; keep it fixed across all images.
        ring = expanded & ~m
        accepted = ring & (dark > 1.5)
        # Guard against extreme expansion on very tiny / irregular masks.
        if int(accepted.sum()) <= 0.30 * max(1, int(m.sum())):
            m = m | accepted
    return m


def refine_final(final: list, image: np.ndarray, variant: str) -> list[np.ndarray]:
    masks = [np.asarray(item[0], dtype=bool) for item in final]
    if variant == 'baseline':
        return masks
    dark = contrast_map(image)
    occupied = np.zeros(image.shape[:2], dtype=bool)
    out: list[np.ndarray] = []
    # Original V4 final masks are score-priority and nonoverlapping. Preserve
    # that order and pixel ownership as closely as possible.
    for mask in masks:
        adjusted = refine_one(mask, dark, variant) & ~occupied
        # An instance remains in output even if an aggressive refinement
        # would empty it; this avoids changing instance count during ablation.
        if not adjusted.any():
            adjusted = mask & ~occupied
        out.append(adjusted)
        occupied |= adjusted
    if len(out) > 1:
        accum = np.zeros(image.shape[:2], dtype=np.uint8)
        for mask in out:
            accum += mask.astype(np.uint8)
        assert int(accum.max()) <= 1, 'Refined masks overlap'
    return out


def load_coco(path: str, cache: dict):
    if path not in cache:
        with open(path, 'r', encoding='utf-8') as f:
            coco = json.load(f)
        imgs = {stable_id(img['id']): img for img in coco.get('images', [])}
        annotations = defaultdict(list)
        for ann in coco.get('annotations', []):
            annotations[stable_id(ann['image_id'])].append(ann)
        cache[path] = (imgs, annotations)
    return cache[path]


def unpack_prediction_output(x):
    if isinstance(x, tuple) and len(x) == 2 and isinstance(x[1], dict):
        return x[0]
    return x


def make_stats_row(stats: PQStats, rec, variant: str) -> dict:
    return dict(
        variant=variant, fold=int(rec['fold']), physical_key=str(rec['physical_key']),
        source_file_name=rec['source_file_name'], gt_instances=int(rec['gt_instances']),
        pred_instances=int(rec['pred_instances']), tp=stats.tp, fp=stats.fp,
        fn=stats.fn, sum_iou=stats.sum_iou, record_pq=stats.pq,
    )


def check_reference_baseline(measured: pd.DataFrame, ref_path: str, tolerance=0.00003):
    ref = pd.read_csv(ref_path)
    base = measured[measured.variant == 'baseline'].reset_index(drop=True)
    if len(ref) != len(base):
        raise RuntimeError(f'Baseline row count mismatch: new {len(base)} vs reference {len(ref)}')
    for col in ('fold', 'physical_key', 'gt_instances', 'tp', 'fp', 'fn'):
        if col not in ref:
            raise RuntimeError(f'Reference baseline missing field: {col}')
        mismatches = (ref[col].astype(str).to_numpy() != base[col].astype(str).to_numpy()).sum()
        if mismatches:
            raise RuntimeError(f'Baseline differs from previous V4 on {col} ({mismatches} records). STOP and check model/config/manifest!')
    if not np.allclose(ref['sum_iou'].to_numpy(), base['sum_iou'].to_numpy(), atol=tolerance, rtol=0):
        raise RuntimeError('Baseline sum_iou values differ from existing V4 baseline; STOP.')
    print('BASELINE CHECK: PASSED, row-by-row TP / FP / FN / IoU match')


def summarise(rows: pd.DataFrame, repeats=1000, seed=2026):
    summary = {}
    sums = {}
    for variant in VARIANTS:
        block = rows[rows.variant == variant]
        tp = int(block.tp.sum()); fp = int(block.fp.sum()); fn = int(block.fn.sum())
        si = float(block.sum_iou.sum()); denom = tp + 0.5 * (fp + fn)
        pq = si / denom if denom else 0.
        summary[variant] = dict(tp=tp, fp=fp, fn=fn, sum_iou=si,
                                pq=pq, sq=si/tp if tp else 0., rq=tp/denom if denom else 0.,
                                records=len(block))
        sums[variant] = block.groupby('physical_key')[['tp','fp','fn','sum_iou']].sum()
    images = sorted(set(rows.physical_key))
    rng = np.random.default_rng(seed)
    # Physical-image cluster bootstrap; repeated annotation records travel together.
    values = {v:sums[v].reindex(images,fill_value=0).to_numpy(dtype=float) for v in VARIANTS}
    b = []
    for _ in range(repeats):
        picks = rng.integers(0,len(images),size=len(images))
        base = values['baseline'][picks].sum(axis=0)
        base_pq=base[3]/(base[0]+0.5*(base[1]+base[2]))
        group=[]
        for v in VARIANTS[1:]:
            stats=values[v][picks].sum(axis=0)
            pq=stats[3]/(stats[0]+0.5*(stats[1]+stats[2]))
            group.append(float(pq-base_pq))
        b.append(group)
    boot=np.array(b)
    for j,v in enumerate(VARIANTS[1:]):
        summary[v]['delta_pq_vs_baseline'] = summary[v]['pq'] - summary['baseline']['pq']
        summary[v]['cluster_bootstrap_delta_ci95'] = [float(x) for x in np.percentile(boot[:,j],[2.5,97.5])]
        basepq = values['baseline'][:,3] / np.maximum(1.0, values['baseline'][:,0]+0.5*(values['baseline'][:,1]+values['baseline'][:,2]))
        curpq = values[v][:,3] / np.maximum(1.0, values[v][:,0]+0.5*(values[v][:,1]+values[v][:,2]))
        diffs=curpq-basepq
        summary[v]['physical_images_improved'] = int((diffs > 1e-10).sum())
        summary[v]['physical_images_worse'] = int((diffs < -1e-10).sum())
        summary[v]['physical_images_equal'] = int((abs(diffs) <= 1e-10).sum())
    return dict(records=len(rows[rows.variant=='baseline']), physical_images=len(images),
                inference='V4 fold-0 one-model OOF, NOT a V4 three-model ensemble',
                method='fixed input-only one-pixel image-guided morphological refinements; GT used only to score PQ',
                bootstrap_repeats=repeats, variants=summary,
                caveat='This fold was used for checkpoint selection and is a development validation set. No Kaggle improvement is guaranteed.')


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--model',required=True)
    p.add_argument('--manifest',required=True)
    p.add_argument('--baseline-csv',required=True)
    p.add_argument('--output-dir',default='v11_2_results')
    p.add_argument('--fold',type=int,default=0)
    p.add_argument('--limit-records',type=int,default=0,help='Smoke test; 0 means all records')
    p.add_argument('--device',default='0')
    p.add_argument('--tile-size',type=int,default=1024)
    p.add_argument('--stride',type=int,default=768)
    p.add_argument('--imgsz',type=int,default=1024)
    p.add_argument('--conf',type=float,default=.30)
    p.add_argument('--max-det-per-tile',type=int,default=30)
    p.add_argument('--merge-iou',type=float,default=.10)
    p.add_argument('--merge-containment',type=float,default=.45)
    p.add_argument('--merge-min-intersection',type=int,default=32)
    p.add_argument('--max-instances',type=int,default=15)
    p.add_argument('--duplicate-iou',type=float,default=.80)
    p.add_argument('--duplicate-containment',type=float,default=.88)
    args=p.parse_args()
    from ultralytics import YOLO
    df = pd.read_csv(args.manifest)
    if 'fold' not in df:
        raise ValueError('Manifest missing fold')
    if not (df.fold.astype(int) == args.fold).all():
        raise ValueError(f'Manifest must contain ONLY fold={args.fold} records; use v8_fold0_manifest.csv')
    if args.limit_records>0:
        df = df.iloc[:args.limit_records].copy()
    out=Path(args.output_dir); out.mkdir(parents=True,exist_ok=True)
    model=YOLO(args.model)
    coco_cache={}; image_index_cache={}
    results=[]
    last_key=None; last_image=None; last_final=None
    for idx,(_,row) in enumerate(df.iterrows(),1):
        path=str(row['source_json']); images_dir=Path(str(row['source_images_dir'])); key=str(row['source_image_id_json'])
        images, anns=load_coco(path,coco_cache)
        img_info=images[key]
        if str(images_dir) not in image_index_cache:
            image_index_cache[str(images_dir)]=build_image_index(images_dir)
        im_path=resolve_image_path(img_info.get('file_name',row.get('source_file_name','')),
                                   image_index_cache[str(images_dir)])
        # Cache adjacent annotation records for the same physical image.
        if str(im_path) == last_key:
            image=last_image; final=last_final
        else:
            image=cv2.imread(str(im_path),cv2.IMREAD_COLOR)
            if image is None:
                raise RuntimeError(f'Cannot read image {im_path}')
            po=predict_native_image(model,image,
                tile_size=args.tile_size,stride=args.stride,imgsz=args.imgsz,
                conf=args.conf,max_det_per_tile=args.max_det_per_tile,
                merge_iou=args.merge_iou,merge_containment=args.merge_containment,
                merge_min_intersection=args.merge_min_intersection,
                max_instances=args.max_instances,duplicate_iou=args.duplicate_iou,
                duplicate_containment=args.duplicate_containment,device=args.device)
            final=unpack_prediction_output(po)
            last_key=str(im_path); last_image=image; last_final=final
        h,w=image.shape[:2]
        gt=[annotation_to_mask(ann,h,w) for ann in anns.get(key,[])]
        for variant in VARIANTS:
            pred=refine_final(final,image,variant)
            stats=panoptic_quality(gt,pred,iou_threshold=0.5)
            rec=dict(fold=args.fold,physical_key=row['physical_key'],
                source_file_name=img_info.get('file_name',row.get('source_file_name','')),
                gt_instances=len(gt),pred_instances=len(pred))
            results.append(make_stats_row(stats,rec,variant))
        if idx % 20 == 0 or idx == len(df):
            print(f'[{idx}/{len(df)}] completed, same YOLO inference for all 4 variants',flush=True)
    results=pd.DataFrame(results)
    if args.limit_records == 0:
        check_reference_baseline(results,args.baseline_csv)
    else:
        print('SMOKE RUN: baseline verification skipped, do not use partial results to choose a variant')
    results.to_csv(out/'record_metrics.csv',index=False)
    summary=summarise(results)
    (out/'summary.json').write_text(json.dumps(summary,indent=2,ensure_ascii=False),encoding='utf-8')
    print('=== V11.2 PQ RESULTS ===')
    for name,s in summary['variants'].items():
        print(f'{name:12s}: PQ {s["pq"]:.5f} TP/FP/FN {s["tp"]}/{s["fp"]}/{s["fn"]}')
    print('Saved:',out/'summary.json')

if __name__=='__main__':
    main()
