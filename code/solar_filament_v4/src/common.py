from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from pycocotools import mask as mask_utils

IMAGE_EXTS = {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff'}


def stable_id(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True)


def physical_image_key(file_name: str) -> str:
    stem = Path(str(file_name)).stem
    stem = re.sub(r'__.*$', '', stem)
    m = re.match(r'^\d{6}-(.+)$', stem)
    if m:
        stem = m.group(1)
    return stem


def build_image_index(images_dir: Path) -> dict[str, Path]:
    index: dict[str, Path] = {}
    for p in images_dir.rglob('*'):
        if p.is_file() and p.suffix.lower() in IMAGE_EXTS:
            index.setdefault(p.name, p)
            index.setdefault(p.stem, p)
            index.setdefault(physical_image_key(p.name), p)
    return index


def resolve_image_path(file_name: str, index: dict[str, Path]) -> Path:
    candidates = [
        str(file_name),
        Path(str(file_name)).name,
        Path(str(file_name)).stem,
        physical_image_key(str(file_name)),
    ]
    for key in candidates:
        if key in index:
            return index[key]
    raise FileNotFoundError(f'Could not resolve image for COCO file_name={file_name!r}')


def annotation_to_mask(annotation: dict, height: int, width: int) -> np.ndarray:
    seg = annotation.get('segmentation')
    if seg is None:
        return np.zeros((height, width), dtype=bool)

    if isinstance(seg, list):
        if len(seg) == 0:
            return np.zeros((height, width), dtype=bool)
        rles = mask_utils.frPyObjects(seg, height, width)
        rle = mask_utils.merge(rles)
        mask = mask_utils.decode(rle)
    elif isinstance(seg, dict):
        rle = dict(seg)
        if isinstance(rle.get('counts'), str):
            rle['counts'] = rle['counts'].encode('utf-8')
        mask = mask_utils.decode(rle)
    else:
        raise TypeError(f'Unsupported segmentation type: {type(seg)}')

    if mask.ndim == 3:
        mask = np.any(mask, axis=2)
    return mask.astype(bool)


def mask_to_single_polygon(mask: np.ndarray, min_points: int = 3) -> list[float] | None:
    mask_u8 = mask.astype(np.uint8)
    contours, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = [c for c in contours if len(c) >= min_points]
    if not contours:
        return None
    areas = [abs(float(cv2.contourArea(c))) for c in contours]
    contour = contours[int(np.argmax(areas))].reshape(-1, 2).astype(np.float32)
    return contour.reshape(-1).tolist()


def encode_mask_to_rle(binary_mask: np.ndarray) -> str:
    mask = np.asfortranarray(binary_mask.astype(np.uint8))
    encoded = mask_utils.encode(mask)
    return encoded['counts'].decode('utf-8')


def decode_rle(rle_string: str, height: int, width: int) -> np.ndarray:
    rle = {'size': [int(height), int(width)], 'counts': str(rle_string).encode('utf-8')}
    return mask_utils.decode(rle).astype(bool)
