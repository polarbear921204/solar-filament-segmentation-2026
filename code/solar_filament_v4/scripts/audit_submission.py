from __future__ import annotations

import argparse, sys
from collections import defaultdict
from pathlib import Path
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT))
from src.common import decode_rle, physical_image_key, IMAGE_EXTS


def main():
    p=argparse.ArgumentParser(); p.add_argument('--submission',required=True); p.add_argument('--test-dir',required=True); p.add_argument('--height',type=int,default=2048); p.add_argument('--width',type=int,default=2048); args=p.parse_args()
    df=pd.read_csv(args.submission); req=['filament_id','segmentation_rle']
    if list(df.columns)!=req: raise RuntimeError(f'Columns must be exactly {req}')
    if df.isna().any().any(): raise RuntimeError('Null values found.')
    if not df['filament_id'].is_unique: raise RuntimeError('Duplicate filament_id values found.')
    grouped=defaultdict(list)
    for _,row in df.iterrows(): grouped[str(row['filament_id']).rsplit('_',1)[0]].append(row)
    invalid=empty=overlap=0
    for image_id,rows in grouped.items():
        occupied=np.zeros((args.height,args.width),dtype=bool)
        for row in rows:
            try: mask=decode_rle(row['segmentation_rle'],args.height,args.width)
            except Exception: invalid+=1; continue
            if mask.shape!=(args.height,args.width): invalid+=1; continue
            if not mask.any(): empty+=1
            if np.any(mask & occupied): overlap+=1
            occupied |= mask
    test_ids={physical_image_key(p.name) for p in Path(args.test_dir).iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTS}; sub_ids=set(grouped)
    missing=sorted(test_ids-sub_ids); extra=sorted(sub_ids-test_ids)
    print('\n=== V3 SUBMISSION AUDIT ==='); print('rows               :',len(df)); print('predicted image ids:',len(sub_ids)); print('test image ids     :',len(test_ids)); print('missing test images:',len(missing)); print('extra image ids    :',len(extra)); print('invalid RLE rows   :',invalid); print('empty masks        :',empty); print('overlap detections :',overlap)
    if missing[:10]: print('first missing:',missing[:10])
    if extra[:10]: print('first extra:',extra[:10])
    if invalid or overlap or extra: raise RuntimeError('AUDIT FAILED')
    print('AUDIT PASSED')


if __name__=='__main__': main()
