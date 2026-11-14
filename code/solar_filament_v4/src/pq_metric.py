from __future__ import annotations

from dataclasses import dataclass
import numpy as np


@dataclass
class PQStats:
    tp: int = 0
    fp: int = 0
    fn: int = 0
    sum_iou: float = 0.0

    def __add__(self, other):
        return PQStats(self.tp + other.tp, self.fp + other.fp, self.fn + other.fn, self.sum_iou + other.sum_iou)

    @property
    def pq(self):
        d = self.tp + 0.5 * self.fp + 0.5 * self.fn
        return self.sum_iou / d if d > 0 else 0.0

    @property
    def sq(self):
        return self.sum_iou / self.tp if self.tp > 0 else 0.0

    @property
    def rq(self):
        d = self.tp + 0.5 * self.fp + 0.5 * self.fn
        return self.tp / d if d > 0 else 0.0


def panoptic_quality(gt_masks, pred_masks, iou_threshold=0.5):
    if not gt_masks:
        return PQStats(fp=len(pred_masks))
    if not pred_masks:
        return PQStats(fn=len(gt_masks))

    candidates = []
    gt_areas = [int(m.sum()) for m in gt_masks]
    pred_areas = [int(m.sum()) for m in pred_masks]
    for i, gt in enumerate(gt_masks):
        for j, pred in enumerate(pred_masks):
            inter = int(np.logical_and(gt, pred).sum())
            if inter == 0:
                continue
            union = gt_areas[i] + pred_areas[j] - inter
            iou = inter / union if union > 0 else 0.0
            if iou > iou_threshold:
                candidates.append((iou, i, j))
    candidates.sort(reverse=True, key=lambda x: x[0])
    used_gt, used_pred = set(), set()
    tp, sum_iou = 0, 0.0
    for iou, i, j in candidates:
        if i in used_gt or j in used_pred:
            continue
        used_gt.add(i); used_pred.add(j)
        tp += 1; sum_iou += iou
    return PQStats(tp=tp, fp=len(pred_masks)-tp, fn=len(gt_masks)-tp, sum_iou=sum_iou)
