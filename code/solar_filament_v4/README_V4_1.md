# V4.1: Cache V4 model predictions and tune fusion offline

The scripts in `scripts/` are a **patch for your existing `solar_filament_v4/`** project.
Copy both `.py` files to `solar_filament_v4/scripts/`.
They reuse your existing `src/` modules. No model retraining is required.

## Why do this?

The V3 test had 1,152 masks, V4 had 1,130; both covered 178 of 180 test images.
That difference alone **does not** reveal how many predictions were supported by one, two, or three fold models.

The cache contains each model's masks **before cross-model matching**. Once saved,
new matching/voting settings can be tested without running YOLO again.

## A. One-time GPU caching

Run from `solar_filament/`:

```bash
python solar_filament_v4/scripts/cache_v4_predictions.py \
  --models \
    ./runs/segment/runs_v4/yolov8m_patch_v4_fold0/weights/best.pt \
    ./runs/segment/runs_v4/yolov8m_patch_v4_fold1/weights/best.pt \
    ./runs/segment/runs_v4/yolov8m_patch_v4_fold2/weights/best.pt \
  --test-dir "/home/ivan/SSD/solar_filament/MAGFiLO_1.0_Kaggle_2026/test/test_images" \
  --tile-size 1024 --stride 768 --imgsz 1024 --conf 0.30 \
  --merge-iou 0.10 --merge-containment 0.45 \
  --max-instances 15 --duplicate-iou 0.80 --duplicate-containment 0.88 \
  --output v4_1_predictions.jsonl
```

If interrupted, repeat the command with `--resume` (all other arguments unchanged).

## B. Reproduce your previous V4 baseline (no GPU inference)

```bash
python solar_filament_v4/scripts/build_ensemble_from_cache.py \
  --cache v4_1_predictions.jsonl \
  --min-model-support 2 --pixel-vote-threshold 0.50 \
  --ensemble-match-iou 0.25 --ensemble-match-containment 0.55 \
  --compare-to submission_v4_ensemble.csv \
  --output submission_v4_1_baseline.csv
```

Check the output:
- `clusters support=1/2/3`: how many cross-model groups have each number of supporters;
- `BASELINE REPRODUCED EXACTLY`: the caching pipeline and prior V4 submission are identical.

If the reproduction is **not** exact, investigate before comparing variants.

## C. Optional offline candidates

1. Permit one-model predictions (may increase false positives):

```bash
python solar_filament_v4/scripts/build_ensemble_from_cache.py \
  --cache v4_1_predictions.jsonl --min-model-support 1 \
  --pixel-vote-threshold 0.50 --output submission_v4_1_support1.csv
```

2. Require stronger agreement on pixels (for a two-model cluster, 0.60 requires both masks to contain a pixel):

```bash
python solar_filament_v4/scripts/build_ensemble_from_cache.py \
  --cache v4_1_predictions.jsonl --min-model-support 2 \
  --pixel-vote-threshold 0.60 --output submission_v4_1_vote060.csv
```

3. Looser cross-model instance matching:

```bash
python solar_filament_v4/scripts/build_ensemble_from_cache.py \
  --cache v4_1_predictions.jsonl --ensemble-match-iou 0.20 \
  --ensemble-match-containment 0.55 \
  --output submission_v4_1_match020.csv
```

**Do not submit all variants automatically.** Kaggle Public LB is not a replacement
for an honest ensemble validation set: of three fold models, two saw a given
training image. Fold-wise OOF PQ measures single held-out models, not the 3-model
ensemble on an unseen image. Ideally train two additional independently seeded
models on the existing V3 **training-only** split, leaving all V3 validation
images untouched, and use those to calibrate ensemble settings on the V3 holdout.

## D. Audit before submission

```bash
python solar_filament_v4/scripts/audit_submission.py \
  --submission submission_v4_1_vote060.csv \
  --test-dir "/home/ivan/SSD/solar_filament/MAGFiLO_1.0_Kaggle_2026/test/test_images" \
  --height 2048 --width 2048
```

Check `invalid RLE rows=0`, `overlap detections=0`, `extra image ids=0`.

**Voting note:** In the current implementation, `pixel_vote_threshold=0.50`
is equivalent to OR for clusters supported by **two** models and majority vote
for clusters supported by **three** models. A value of 0.40 behaves the same on
both two- and three-member clusters, so don't waste a run on 0.40.
