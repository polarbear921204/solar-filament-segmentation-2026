#!/usr/bin/env bash
# reproduce V4 then apply the frozen final V11.3 Trim1. DOES NOT overwrite 0.39 original.
set -euo pipefail
if [[ $# -ne 4 ]]; then
  echo 'Usage: bash scripts/run_final_pipeline.sh TEST_DIR FOLD0_BEST_PT FOLD1_BEST_PT FOLD2_BEST_PT' >&2
  exit 2
fi
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TEST_DIR="$(realpath "$1")"; M0="$(realpath "$2")"; M1="$(realpath "$3")"; M2="$(realpath "$4")"
mkdir -p "$ROOT/outputs"
python "$ROOT/code/solar_filament_v4/scripts/inference_ensemble_test.py" \
  --models "$M0" "$M1" "$M2" --test-dir "$TEST_DIR" \
  --tile-size 1024 --stride 768 --imgsz 1024 --conf 0.30 \
  --merge-iou 0.10 --merge-containment 0.45 --max-instances 15 \
  --duplicate-iou 0.80 --duplicate-containment 0.88 \
  --ensemble-match-iou 0.25 --ensemble-match-containment 0.55 \
  --min-model-support 2 --pixel-vote-threshold 0.50 \
  --device 0 --output "$ROOT/outputs/submission_v4_ensemble.csv"
python "$ROOT/code/v11_3_trim_submission/make_trim1_submission.py" \
  --submission "$ROOT/outputs/submission_v4_ensemble.csv" --test-dir "$TEST_DIR" \
  --output "$ROOT/outputs/submission_v11_3_trim1.csv" \
  --summary "$ROOT/outputs/v11_3_trim1_audit.json" \
  --per-image "$ROOT/outputs/v11_3_trim1_images.csv"
python "$ROOT/code/solar_filament_v4/scripts/audit_submission.py" \
  --submission "$ROOT/outputs/submission_v11_3_trim1.csv" --test-dir "$TEST_DIR" \
  --height 2048 --width 2048
sha256sum "$ROOT/outputs/submission_v4_ensemble.csv" "$ROOT/outputs/submission_v11_3_trim1.csv"
cat <<'END'
REFERENCE SHA256 V4:  e6f8b6053d5afe0c0d61374e47ed7abed92da186054819a46e3026ae943849f3
REFERENCE SHA256 V11.3: 13c19935e917018fe2a5c550927c2dc1fdf317987be4dba563e22e756d217f8f
If hashes differ, inspect environment, model ordering, file naming and input test images.
END
