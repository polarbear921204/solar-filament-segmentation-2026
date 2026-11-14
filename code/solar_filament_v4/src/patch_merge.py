from __future__ import annotations

from dataclasses import dataclass
import numpy as np


@dataclass
class GlobalInstance:
    mask: np.ndarray
    score: float
    support: int = 1


def _metrics(a: np.ndarray, b: np.ndarray):
    inter = int(np.logical_and(a, b).sum())
    if inter == 0:
        return 0.0, 0.0, 0
    area_a = int(a.sum())
    area_b = int(b.sum())
    union = area_a + area_b - inter
    iou = inter / union if union > 0 else 0.0
    containment = inter / min(area_a, area_b) if min(area_a, area_b) > 0 else 0.0
    return iou, containment, inter


def merge_patch_instances(masks, scores, *, merge_iou=0.15, merge_containment=0.45, min_intersection=32):
    if not masks:
        return []
    order = np.argsort(np.asarray(scores))[::-1]
    merged = []
    for idx in order:
        mask = np.asarray(masks[int(idx)], dtype=bool)
        score = float(scores[int(idx)])
        if not mask.any():
            continue
        best = None
        best_value = -1.0
        for j, inst in enumerate(merged):
            iou, containment, inter = _metrics(mask, inst.mask)
            if inter < min_intersection:
                continue
            if not (iou >= merge_iou or containment >= merge_containment):
                continue
            value = max(iou, containment)
            if value > best_value:
                best_value = value
                best = j
        if best is None:
            merged.append(GlobalInstance(mask=mask.copy(), score=score, support=1))
        else:
            inst = merged[best]
            inst.mask |= mask
            inst.score = max(inst.score, score)
            inst.support += 1
    return merged
