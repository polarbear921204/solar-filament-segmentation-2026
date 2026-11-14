# Draft for a FOUR-PAGE Technical Report — NOT an official formatted submission

**Proposed title:** Image-Guided Boundary Trimming for Three-Fold YOLOv8m Solar Filament Instance Segmentation

**Authors and affiliations:** [TEAM TO FILL]

> This is editable scientific prose for the teammate. Paste and adapt in the competition's required Overleaf template, check exact page count, and add appropriate literature citations.

## Abstract

We address instance-level solar filament segmentation in 2048×2048 GONG H-alpha observations from the MAGFiLO dataset. Our pipeline combines full-resolution overlapping tiles with three physically grouped YOLOv8m-seg models. Each model predicts instance masks on 1024×1024 tiles; tile-level predictions are merged in original-image coordinates and then fused across models with confidence-ordered instance association, support-based voting, and exclusive pixel ownership. A lightweight image-guided boundary refinement is applied *after* ensemble fusion. This refinement trims selected one-pixel mask boundary regions that are locally brighter than their surroundings, without changing the instance count. The V4 ensemble achieved a displayed Kaggle Public Leaderboard score of 0.38; the same pipeline with the proposed Trim1 operation achieved 0.39. In three-fold **single-model out-of-fold** development evaluation, the same refinement increased pooled Panoptic Quality from 0.425718 to 0.429275. These validation measurements must not be interpreted as a leakage-free evaluation of the three-model ensemble.

## 1. Problem, Data, and Evaluation

Solar filaments are dark, elongated chromospheric structures whose thin and fragmented appearance makes pixel-accurate delineation difficult. The challenge requires a separate segmentation mask for every instance rather than a single foreground/background region. We use 707 physical training observations represented by 1,154 COCO annotation records and 180 held-out test observations. Multiple annotation records for a physical image are grouped to prevent train/validation leakage across expert views.

Predictions are scored using Panoptic Quality (PQ):

$$PQ=\frac{\sum_{(g,p)\in TP}IoU(g,p)}{|TP|+0.5|FP|+0.5|FN|}.$$

The final submission stores the compressed COCO RLE *counts* string for each 2048×2048 predicted filament, together with its unique `filament_id`.

## 2. Physically Grouped Patch-Based Training

We randomly assign physical-image identifiers to three folds (seeds for model training: 42, 43, and 44), ensuring that duplicate or multi-expert annotation records of the same observation remain within a single fold. The full-resolution 2048×2048 images are cropped into overlapping 1024×1024 patches with stride 768. Visible mask fragments are retained when they contain at least 16 pixels and at least 2% of the source instance area. Each visible instance is represented by a normalized YOLO polygon. Three pretrained YOLOv8m-seg models are trained independently, one per held-out fold, for up to 60 epochs with batch size 4 and early-stopping patience 20. Augmentations include rotations up to 180°, small translations and scaling, horizontal/vertical reflections, and limited brightness jitter. Mosaic and mixup are disabled.

## 3. Native-Resolution Inference and Ensemble Fusion

At inference time, each model receives the same nine overlapping 1024×1024 tiles per 2048×2048 image (`imgsz=1024`, confidence threshold 0.30). Its patch masks are mapped back to global coordinates and greedily merged if they share at least 32 pixels and satisfy either IoU ≥ 0.10 or containment ≥ 0.45. Candidate duplicates are suppressed, and up to 15 instances are retained per image for each model.

Next, predicted instances are greedily associated across the three models. Associations require at least 32 intersecting pixels, with IoU ≥ 0.25 or containment ≥ 0.55. Clusters must be supported by at least two distinct models and can contain at most one instance from each model. Pixel voting uses a vote threshold `max(1, 0.5 × support)`: for two-member clusters, a one-vote union; for three-member clusters, at least two votes. Final deduplication and confidence-ordered exclusive pixel assignment prevent mask overlaps. The resulting V4 ensemble formed the original 0.38 Public LB baseline.

## 4. Image-Guided One-Pixel Boundary Trimming (Trim1)

The final improvement does not alter the trained networks. After ensemble fusion and exclusive mask assignment, each test image is converted to grayscale, with Gaussian-smoothed versions `fine = Gaussian(I, σ=1)` and `coarse = Gaussian(I, σ=5)`. The local contrast signal is `dark = coarse − fine`. For each instance mask `M`, a one-pixel rim is estimated by subtracting a 3×3 elliptical erosion from `M`. We remove rim pixels where `dark < −1.5`, but only when the total proposed removal does not exceed 15% of the original mask area. An empty result reverts to the original mask. Consequently, Trim1 only removes existing pixels and cannot introduce new inter-instance overlaps. This input-only post-processing step seeks to reject mask boundary pixels that are inconsistent with dark filament morphology.

## 5. Evaluation and Ablation

| Held-out fold | Original single-model OOF PQ | OOF PQ + Trim1 |
|---|---:|---:|
| 0 | 0.416194 | 0.421691 |
| 1 | 0.435566 | 0.436930 |
| 2 | 0.425271 | 0.429092 |
| **Pooled** | **0.425718** | **0.429275** |

The pooled figures combine fold-level TP, FP, FN, and summed IoUs before computing PQ rather than simply averaging fold PQs. The pooled 1,154-record development comparison shows a +0.003557 absolute PQ change. These OOF results use **one fold-specific model per validation record**, not three-model inference; our three-model ensemble improvement is evidenced separately by the public test leaderboard, which displayed 0.38 without Trim1 and 0.39 with Trim1. The final submission retained 1,130 instances across 178 of the 180 test images. Its audit recorded 1,124 changed masks, 168,390 removed pixels, no empty masks, and zero overlapping output pixels.

## 6. Discussion, Limitations, and Reproducibility

Qualitative diagnostics suggest that filament false negatives often involve small structures or insufficiently localized masks. Trim1 improves boundary precision for some predictions, but it cannot recover a filament that was never detected, and aggressive trimming could remove true filament pixels. The MAGFiLO annotation protocol also permits multiple expert views of a single physical image, potentially leading to differing per-record judgments. Public leaderboard scores are rounded and are not interchangeable with private test performance. The source repository contains fold generation, patch conversion, training, validation, tile merging, ensemble fusion, mask exclusivity, Trim1, and final COCO-RLE export. The precise environment versions, training seeds, model weight checksums, and executable inference commands are recorded for reproducibility. [TEAM: add measured inference runtime, resource requirements, model distribution/licensing, and final source URL.]

## References to add in the official report

- Competition website and official self-evaluation notebook (PQ implementation).
- Ultralytics YOLOv8-seg software citation/license.
- Panoptic Segmentation / Panoptic Quality (Kirillov et al., CVPR 2019).
- MAGFiLO / GONG dataset references and acknowledgement (verify dataset-required citation).

**Important caution:** Do not claim a 0.429275 leakage-free *ensemble* PQ or a verified Private LB of 0.39. Avoid attributing an exact +0.01 underlying PQ change to rounded leaderboard numbers.
