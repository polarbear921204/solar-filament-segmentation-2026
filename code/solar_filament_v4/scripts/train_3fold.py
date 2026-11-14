from __future__ import annotations

import argparse
from ultralytics import YOLO


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--fold', type=int, required=True, choices=[0, 1, 2])
    p.add_argument('--data-root', default='dataset_v4_folds')
    p.add_argument('--model', default='yolov8m-seg.pt')
    p.add_argument('--epochs', type=int, default=60)
    p.add_argument('--patience', type=int, default=20)
    p.add_argument('--imgsz', type=int, default=1024)
    p.add_argument('--batch', type=int, default=4)
    p.add_argument('--device', default='0')
    p.add_argument('--project', default='runs_v4')
    return p.parse_args()


def main():
    args = parse_args()
    model = YOLO(args.model)
    model.train(
        data=f'{args.data_root}/fold{args.fold}/dataset.yaml',
        epochs=args.epochs,
        patience=args.patience,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        project=args.project,
        name=f'yolov8m_patch_v4_fold{args.fold}',
        single_cls=True,
        seed=42 + args.fold,
        deterministic=True,
        degrees=180.0,
        translate=0.03,
        scale=0.08,
        fliplr=0.5,
        flipud=0.5,
        mosaic=0.0,
        mixup=0.0,
        hsv_h=0.0,
        hsv_s=0.0,
        hsv_v=0.12,
        overlap_mask=False,
        cos_lr=True,
        weight_decay=0.0005,
        verbose=True,
    )


if __name__ == '__main__':
    main()
