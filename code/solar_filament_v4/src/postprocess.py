from __future__ import annotations

import numpy as np


def _mask_iou(a: np.ndarray, b: np.ndarray) -> tuple[float, int]:
    inter = int(np.logical_and(a, b).sum())
    if inter == 0:
        return 0.0, 0
    union = int(a.sum()) + int(b.sum()) - inter
    return (inter / union if union > 0 else 0.0), inter


def suppress_duplicates_and_make_exclusive(
    masks,
    scores,
    *,
    max_instances=12,
    min_area=0,
    duplicate_iou=0.80,
    duplicate_containment=0.88,
    duplicate_min_area_ratio=0.50,
):
    if len(masks) == 0:
        return []

    scores = np.asarray(scores, dtype=np.float32)
    order = np.argsort(scores)[::-1]
    kept = []

    for idx in order:
        mask = np.asarray(masks[int(idx)], dtype=bool)
        area = int(mask.sum())
        score = float(scores[int(idx)])
        if area <= min_area:
            continue

        duplicate = False
        for prev_mask, _, prev_area in kept:
            iou, inter = _mask_iou(mask, prev_mask)
            if inter == 0:
                continue
            smaller = min(area, prev_area)
            larger = max(area, prev_area)
            containment = inter / smaller if smaller > 0 else 0.0
            area_ratio = smaller / larger if larger > 0 else 0.0
            if iou >= duplicate_iou or (containment >= duplicate_containment and area_ratio >= duplicate_min_area_ratio):
                duplicate = True
                break

        if not duplicate:
            kept.append((mask, score, area))
        if max_instances > 0 and len(kept) >= max_instances:
            break

    if not kept:
        return []

    occupied = np.zeros_like(kept[0][0], dtype=bool)
    final = []
    for mask, score, _ in kept:
        clean = mask & (~occupied)
        if int(clean.sum()) <= min_area:
            continue
        final.append((clean, score))
        occupied |= clean

    if len(final) > 1:
        total = np.zeros_like(final[0][0], dtype=np.uint16)
        for mask, _ in final:
            total += mask.astype(np.uint16)
        if int(total.max()) > 1:
            raise RuntimeError('Overlapping masks remain after post-processing.')
    return final
