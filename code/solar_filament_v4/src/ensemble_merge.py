from __future__ import annotations

from dataclasses import dataclass
import numpy as np


@dataclass
class EnsembleCluster:
    members: list[tuple[int, np.ndarray, float]]


def _metrics(a: np.ndarray, b: np.ndarray):
    inter = int(np.logical_and(a, b).sum())
    if inter == 0:
        return 0.0, 0.0, 0

    aa = int(a.sum())
    bb = int(b.sum())
    union = aa + bb - inter

    iou = inter / union if union > 0 else 0.0
    containment = inter / min(aa, bb) if min(aa, bb) > 0 else 0.0

    return iou, containment, inter


def _normalize_prediction_item(item):
    """
    Normalize one prediction item to (mask, score).

    Supported:
      (mask, score)
      (mask, score, ...)
      raw mask
    """
    if isinstance(item, (tuple, list)):
        if len(item) == 0:
            return None
        mask = item[0]
        score = float(item[1]) if len(item) >= 2 else 1.0
    else:
        mask = item
        score = 1.0

    return np.asarray(mask, dtype=bool), score


def cluster_instances_across_models(
    model_predictions,
    *,
    match_iou: float = 0.25,
    match_containment: float = 0.55,
    min_intersection: int = 32,
):
    """
    Greedy cross-model clustering.
    At most one instance from each model joins a cluster.
    """

    all_items = []

    for model_idx, preds in enumerate(model_predictions):
        for item in preds:
            normalized = _normalize_prediction_item(item)
            if normalized is None:
                continue

            mask, score = normalized

            if not mask.any():
                continue

            all_items.append(
                (float(score), model_idx, mask)
            )

    all_items.sort(
        reverse=True,
        key=lambda x: x[0],
    )

    clusters = []

    for score, model_idx, mask in all_items:
        best_cluster = None
        best_value = -1.0

        for ci, cluster in enumerate(clusters):
            # Do not allow two instances from the same model in one cluster.
            if any(
                existing_model_idx == model_idx
                for existing_model_idx, _, _ in cluster.members
            ):
                continue

            union_mask = np.zeros_like(mask, dtype=bool)
            for _, cluster_mask, _ in cluster.members:
                union_mask |= cluster_mask

            iou, containment, inter = _metrics(
                mask,
                union_mask,
            )

            if inter < min_intersection:
                continue

            if not (
                iou >= match_iou
                or containment >= match_containment
            ):
                continue

            value = max(iou, containment)

            if value > best_value:
                best_value = value
                best_cluster = ci

        if best_cluster is None:
            clusters.append(
                EnsembleCluster(
                    members=[
                        (model_idx, mask.copy(), score)
                    ]
                )
            )
        else:
            clusters[best_cluster].members.append(
                (model_idx, mask.copy(), score)
            )

    return clusters


def cluster_to_mask(
    cluster,
    *,
    min_model_support: int = 2,
    pixel_vote_threshold: float = 0.50,
):
    model_ids = {
        model_idx
        for model_idx, _, _ in cluster.members
    }

    if len(model_ids) < min_model_support:
        return None

    shape = cluster.members[0][1].shape
    vote = np.zeros(shape, dtype=np.float32)

    score_sum = 0.0

    for _, mask, score in cluster.members:
        vote += mask.astype(np.float32)
        score_sum += float(score)

    support = len(cluster.members)

    threshold = max(
        1.0,
        pixel_vote_threshold * support,
    )

    final_mask = vote >= threshold

    if not final_mask.any():
        return None

    avg_score = score_sum / support

    return final_mask, avg_score, len(model_ids)


def ensemble_instances(
    model_predictions,
    *,
    match_iou: float = 0.25,
    match_containment: float = 0.55,
    min_intersection: int = 32,
    min_model_support: int = 2,
    pixel_vote_threshold: float = 0.50,
):
    clusters = cluster_instances_across_models(
        model_predictions,
        match_iou=match_iou,
        match_containment=match_containment,
        min_intersection=min_intersection,
    )

    fused = []

    for cluster in clusters:
        item = cluster_to_mask(
            cluster,
            min_model_support=min_model_support,
            pixel_vote_threshold=pixel_vote_threshold,
        )

        if item is not None:
            fused.append(item)

    return fused
