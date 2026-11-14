#!/usr/bin/env python3
from pathlib import Path
import hashlib, argparse
REQUIRED = {
    'submission_v11_3_trim1.csv':'13c19935e917018fe2a5c550927c2dc1fdf317987be4dba563e22e756d217f8f',
    'reference/submission_v4_ensemble_LB038.csv':'e6f8b6053d5afe0c0d61374e47ed7abed92da186054819a46e3026ae943849f3',
}
OPTIONAL = {
    'runs/segment/runs_v4/yolov8m_patch_v4_fold0/weights/best.pt':'e67c9ead2bee638efefa0a60d62da8cb095c3719bb059a577790522fb53fde0f',
    'runs/segment/runs_v4/yolov8m_patch_v4_fold1/weights/best.pt':'77fdaa9f9e2aab82af1e8d6c0d9f812ca197b124fbafc0ebd5c3f8ae1965d73c',
    'runs/segment/runs_v4/yolov8m_patch_v4_fold2/weights/best.pt':'4a47bd53a157457230395a719eb49fc863dd3edf544f9fce13a5cc03d331b38b',
}
p=argparse.ArgumentParser(); p.add_argument('--project-root',default='.'); a=p.parse_args()
root=Path(a.project_root)
for rel,checksum in REQUIRED.items():
    f=root/rel
    if not f.exists(): print('MISSING',rel);continue
    h=hashlib.sha256()
    with f.open('rb') as inp:
        for block in iter(lambda:inp.read(4<<20),b''):h.update(block)
    print('PASS' if h.hexdigest()==checksum else 'DIFFERENT',' ',rel,'sha256',h.hexdigest())
for rel,checksum in OPTIONAL.items():
    f=root/rel
    if not f.exists():
        print('OPTIONAL (not included)', rel)
        continue
    h=hashlib.sha256()
    with f.open('rb') as inp:
        for block in iter(lambda:inp.read(4<<20),b''):h.update(block)
    print('PASS' if h.hexdigest()==checksum else 'DIFFERENT',' ',rel,'sha256',h.hexdigest())
