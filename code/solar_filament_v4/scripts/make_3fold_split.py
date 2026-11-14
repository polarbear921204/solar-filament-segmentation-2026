from __future__ import annotations

import argparse
import numpy as np
import pandas as pd


def parse_args():
    p = argparse.ArgumentParser(description='Create grouped 3-fold assignments by physical image.')
    p.add_argument('--manifest', default='dataset_yolo_v2_native/manifest.csv')
    p.add_argument('--output', default='dataset_yolo_v4_3fold_manifest.csv')
    p.add_argument('--seed', type=int, default=42)
    return p.parse_args()


def main():
    args = parse_args()
    df = pd.read_csv(args.manifest)
    physical = np.array(sorted(df['physical_key'].astype(str).unique()))
    rng = np.random.default_rng(args.seed)
    rng.shuffle(physical)

    fold_map = {key: i % 3 for i, key in enumerate(physical)}
    df['fold'] = df['physical_key'].astype(str).map(fold_map)

    if df.groupby('physical_key')['fold'].nunique().max() != 1:
        raise RuntimeError('Physical-image leakage across folds detected.')

    df.to_csv(args.output, index=False)

    print('\n=== 3-FOLD SUMMARY ===')
    print('total physical imgs:', df['physical_key'].nunique())
    for fold in range(3):
        x = df[df['fold'] == fold]
        print(f'fold {fold}: records={len(x)}, physical_imgs={x["physical_key"].nunique()}')
    print('saved:', args.output)


if __name__ == '__main__':
    main()
