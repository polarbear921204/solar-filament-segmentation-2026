# Solar Filament Segmentation — V4 3-Fold Ensemble

V4 builds on the successful V3 native-resolution patch pipeline.

Your current reference is approximately:

```text
V2 Public LB: 0.35
V3 Public LB: 0.37
V3 local PQ : ~0.423
```

V4 keeps the proven V3 geometry and post-processing:

```text
tile_size = 1024
stride = 768
imgsz = 1024
conf = 0.30
merge_iou = 0.10
merge_containment = 0.45
max_instances = 15
duplicate_iou = 0.80
duplicate_containment = 0.88
```

The new variable is model diversity from three physical-image-grouped folds.

## Important validation note

A standard 3-fold CV ensemble has a subtle leakage trap: for any training image, two of
three fold models have seen it. Therefore **do not evaluate the three-model ensemble on
all training records and call that local PQ**.

V4 instead provides `evaluate_oof_pq.py`, which scores every image only with the fold
model that did NOT train on that physical image. This is an honest OOF estimate of the
three fold models. The actual three-model fusion is used on the unseen Kaggle test set.

## Layout

```text
solar_filament_v4/
├── README_V4.md
├── requirements_v4.txt
├── src/
│   ├── common.py
│   ├── tiling.py
│   ├── patch_merge.py
│   ├── postprocess.py
│   ├── infer_native.py
│   ├── ensemble_merge.py
│   └── pq_metric.py
└── scripts/
    ├── make_3fold_split.py
    ├── prepare_fold_patches.py
    ├── train_3fold.py
    ├── evaluate_oof_pq.py
    ├── inference_ensemble_test.py
    └── audit_submission.py
```

## Step 1 — make the grouped 3-fold split

```bash
python solar_filament_v4/scripts/make_3fold_split.py \
  --manifest dataset_yolo_v2_native/manifest.csv \
  --output dataset_yolo_v4_3fold_manifest.csv
```

Expected: 707 unique physical images split roughly 236 / 236 / 235, with all records
for the same physical image assigned to the same fold.

## Step 2 — prepare patches for all three folds

```bash
python solar_filament_v4/scripts/prepare_fold_patches.py \
  --manifest dataset_yolo_v4_3fold_manifest.csv \
  --fold 0 --output-root dataset_v4_folds \
  --tile-size 1024 --stride 768

python solar_filament_v4/scripts/prepare_fold_patches.py \
  --manifest dataset_yolo_v4_3fold_manifest.csv \
  --fold 1 --output-root dataset_v4_folds \
  --tile-size 1024 --stride 768

python solar_filament_v4/scripts/prepare_fold_patches.py \
  --manifest dataset_yolo_v4_3fold_manifest.csv \
  --fold 2 --output-root dataset_v4_folds \
  --tile-size 1024 --stride 768
```

## Step 3 — train the three fold models

```bash
python solar_filament_v4/scripts/train_3fold.py \
  --fold 0 --data-root dataset_v4_folds \
  --epochs 60 --patience 20 --imgsz 1024 --batch 4 --device 0 --project runs_v4

python solar_filament_v4/scripts/train_3fold.py \
  --fold 1 --data-root dataset_v4_folds \
  --epochs 60 --patience 20 --imgsz 1024 --batch 4 --device 0 --project runs_v4

python solar_filament_v4/scripts/train_3fold.py \
  --fold 2 --data-root dataset_v4_folds \
  --epochs 60 --patience 20 --imgsz 1024 --batch 4 --device 0 --project runs_v4
```

Find the actual weight paths with:

```bash
find . -type f -path "*yolov8m_patch_v4_fold*/weights/best.pt" -print
```

## Step 4 — honest OOF evaluation

Replace the three paths below with the actual `best.pt` files:

```bash
python solar_filament_v4/scripts/evaluate_oof_pq.py \
  --models FOLD0_BEST_PT FOLD1_BEST_PT FOLD2_BEST_PT \
  --manifest dataset_yolo_v4_3fold_manifest.csv \
  --tile-size 1024 --stride 768 --imgsz 1024 --conf 0.30 \
  --max-instances 15 \
  --merge-iou 0.10 --merge-containment 0.45 \
  --duplicate-iou 0.80 --duplicate-containment 0.88 \
  --output eval_v4_oof.csv
```

This reports per-fold PQ and aggregate honest OOF PQ.

Do not expect OOF PQ to equal the V3 fixed-holdout PQ exactly because the validation
composition is different. What matters is that all three fold models are reasonably
consistent and none collapses badly.

## Step 5 — ensemble inference on the Kaggle test set

Baseline V4 ensemble settings:

```text
ensemble_match_iou         = 0.25
ensemble_match_containment = 0.55
min_model_support          = 2
pixel_vote_threshold       = 0.50
```

Run:

```bash
python solar_filament_v4/scripts/inference_ensemble_test.py \
  --models FOLD0_BEST_PT FOLD1_BEST_PT FOLD2_BEST_PT \
  --test-dir "/home/ivan/SSD/solar_filament/MAGFiLO_1.0_Kaggle_2026/test/test_images" \
  --tile-size 1024 --stride 768 --imgsz 1024 --conf 0.30 \
  --max-instances 15 \
  --merge-iou 0.10 --merge-containment 0.45 \
  --duplicate-iou 0.80 --duplicate-containment 0.88 \
  --ensemble-match-iou 0.25 \
  --ensemble-match-containment 0.55 \
  --min-model-support 2 \
  --pixel-vote-threshold 0.50 \
  --output submission_v4_ensemble.csv
```

Fusion logic:

```text
fold0 prediction ─┐
fold1 prediction ─┼─> cross-model instance matching
fold2 prediction ─┘            ↓
                      require >= 2 models
                               ↓
                         pixel voting
                               ↓
                 duplicate suppression/non-overlap
                               ↓
                         final instances
```

## Step 6 — audit before Kaggle

```bash
python solar_filament_v4/scripts/audit_submission.py \
  --submission submission_v4_ensemble.csv \
  --test-dir "/home/ivan/SSD/solar_filament/MAGFiLO_1.0_Kaggle_2026/test/test_images" \
  --height 2048 --width 2048
```

Require:

```text
invalid RLE rows   : 0
overlap detections : 0
extra image ids    : 0
AUDIT PASSED
```

## First tuning after the baseline submission

Because honest local evaluation of a standard CV test ensemble is not available without
leakage, do not aggressively tune ensemble thresholds against the training set.
Start with the baseline above. If the Kaggle result is promising, the safest next tests are
small, interpretable changes:

```text
min_model_support = 2   (baseline)
ensemble_match_iou = 0.20 or 0.30
pixel_vote_threshold = 0.40 or 0.60
```

Use only a small number of Kaggle submissions for these comparisons.
