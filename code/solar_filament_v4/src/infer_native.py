from __future__ import annotations

import cv2
import numpy as np

from .tiling import make_tiles
from .patch_merge import merge_patch_instances
from .postprocess import suppress_duplicates_and_make_exclusive


def predict_native_image(
    model,
    image_bgr,
    *,
    tile_size=1024,
    stride=768,
    imgsz=1024,
    conf=0.30,
    max_det_per_tile=30,
    merge_iou=0.15,
    merge_containment=0.45,
    merge_min_intersection=32,
    max_instances=12,
    min_area=0,
    duplicate_iou=0.80,
    duplicate_containment=0.88,
    device='0',
):
    h, w = image_bgr.shape[:2]
    global_masks, global_scores = [], []
    tiles = make_tiles(w, h, tile_size=tile_size, stride=stride)

    for tile in tiles:
        patch = image_bgr[tile.y0:tile.y1, tile.x0:tile.x1]
        result = model.predict(source=patch, imgsz=imgsz, conf=conf, retina_masks=True, max_det=max_det_per_tile, device=device, verbose=False)[0]
        if result.masks is None or len(result.masks.data) == 0:
            continue
        masks = result.masks.data.cpu().numpy()
        scores = result.boxes.conf.cpu().numpy().astype(float) if result.boxes is not None and result.boxes.conf is not None else np.ones(len(masks), dtype=np.float32)
        for mask, score in zip(masks, scores):
            ph, pw = tile.y1 - tile.y0, tile.x1 - tile.x0
            if mask.shape != (ph, pw):
                mask = cv2.resize(mask.astype(np.float32), (pw, ph), interpolation=cv2.INTER_NEAREST)
            local = mask > 0.5
            if not local.any():
                continue
            g = np.zeros((h, w), dtype=bool)
            g[tile.y0:tile.y1, tile.x0:tile.x1] = local
            global_masks.append(g)
            global_scores.append(float(score))

    merged = merge_patch_instances(global_masks, global_scores, merge_iou=merge_iou, merge_containment=merge_containment, min_intersection=merge_min_intersection)
    final = suppress_duplicates_and_make_exclusive(
        [m.mask for m in merged], [m.score for m in merged],
        max_instances=max_instances, min_area=min_area,
        duplicate_iou=duplicate_iou, duplicate_containment=duplicate_containment,
    )
    return final, {
        'tiles': len(tiles),
        'raw_patch_instances': len(global_masks),
        'merged_instances': len(merged),
        'final_instances': len(final),
    }
