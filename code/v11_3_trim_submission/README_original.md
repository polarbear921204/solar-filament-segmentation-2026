# V11.3：V4 三模型 Ensemble 最終 Masks 的 Trim 1px

## 目標

在不修改原有 Kaggle Public LB 0.38 的 V4 三模型模型、推論、ensemble、instance IDs 的前提下，對 **已經產生的** `submission_v4_ensemble.csv` 做一次與 V11.2 完全相同的 **trim1** 處理，輸出候選 `submission_v11_3_trim1.csv`。

**不用 GPU、不需要模型權重、不會重新訓練。** 只需原始 2048×2048 Test 影像、舊 submission，以及 `opencv-python`、`pandas`、`numpy`、`pycocotools`。

此腳本逐張讀取 Test 影像，針對每個 compressed COCO RLE mask 依照 V11.2 規則修剪：

1. `dark = GaussianBlur(gray, sigma=5) - GaussianBlur(gray, sigma=1)`。
2. 選取 3×3 橢圓 kernel 侵蝕後的最外圍 1px。
3. 僅修剪 `dark < -1.5` 的邊緣像素；若刪除量超過該 mask 的 15%，則保留原 mask。
4. **不補入任何新像素**，保持 instance ID、instance 數量和原本非重疊性。

## 實驗依據（V11.2 單模型 OOF，非三模型 Ensemble）

| Fold | Baseline PQ | trim1 PQ | Delta |
|---|---:|---:|---:|
| 0 | 0.416194 | 0.421691 | +0.005497 |
| 1 | 0.435566 | 0.436930 | +0.001364 |
| 2 | 0.425271 | 0.429092 | +0.003822 |
| **Pooled** | **0.425718** | **0.429275** | **+0.003557** |

完整加總數值在 `pooled_oof_summary.json`。Pooled PQ 是以全部 TP/FP/FN/sum_iou 重新計算，**不是三個 fold PQ 的算術平均**。這批 validation 已用於各 fold checkpoint 選擇，而且單模型 OOF 不代表最終三模型 Ensemble 一定改善。尚未驗證 Kaggle Test PQ。

## Linux 執行順序

在專案根目錄，例如：

```bash
cd /home/ivan/SSD/solar_filament
# 將 ZIP 放在此目錄後解壓：
unzip -o v11_3_trim_submission.zip
# 檢查程式實際所在位置；若 ZIP 不小心被放進同名資料夾，再依 find 結果調整：
find . -name make_trim1_submission.py
```

### A. 先跑三張影像的 Smoke Test

```bash
python v11_3_trim_submission/make_trim1_submission.py \
  --submission submission_v4_ensemble.csv \
  --test-dir MAGFiLO_1.0_Kaggle_2026/test/test_images \
  --limit-images 3 \
  --output v11_3_smoke_INCOMPLETE.csv \
  --summary v11_3_smoke_audit.json \
  --per-image v11_3_smoke_images.csv
```

**Smoke Test 產生的 CSV 不完整，禁止提交 Kaggle。**

### B. 全部 178 張原本有 mask 的 Test 影像

```bash
python v11_3_trim_submission/make_trim1_submission.py \
  --submission submission_v4_ensemble.csv \
  --test-dir MAGFiLO_1.0_Kaggle_2026/test/test_images \
  --output submission_v11_3_trim1.csv \
  --summary v11_3_trim1_audit.json \
  --per-image v11_3_trim1_images.csv
```

預期：`mask_rows_input == mask_rows_output == 1130`、`images_with_predictions == 178`、`number_of_output_overlaps == 0`，且原始 V4 CSV 保持不變。`masks_changed` 一般應大於零；若是零，先確認影像路徑與演算法是否真正作用。

### C. 使用原有 V4 Audit 再核對一次

```bash
python solar_filament_v4/scripts/audit_submission.py \
  --submission submission_v11_3_trim1.csv \
  --test-dir MAGFiLO_1.0_Kaggle_2026/test/test_images
```

應顯示 `AUDIT PASSED`、`overlap detections : 0`、`invalid RLE rows : 0`、`extra image ids : 0`。先前有 **兩張 Test 影像完全沒有預測**，因此 `missing test images : 2` 通常是預期狀態，不是自動失敗。

### D. 保留原始參考分數

```bash
sha256sum submission_v4_ensemble.csv submission_v11_3_trim1.csv
```

兩個 SHA 通常不同（除非修剪完全沒修改任何 mask）。這是正常的。**請勿覆寫 `submission_v4_ensemble.csv`。**

## 跑完後提供

- `v11_3_trim1_audit.json`（用來確認變動範圍及安全性）
- `v11_3_trim1_images.csv`（可以分析哪些影像受影響）
- 最後 Audit 的 Terminal 輸出

經過檢查後，如果競賽仍允許提交，可以把 `submission_v11_3_trim1.csv` 作為 **單次有依據的新候選提交**，但不要再在公開榜單上無止境微調 1px 門檻。保留 Kaggle 0.38 的 V4 原版以便復原。

## 注意

- 這裡是 **trim after three-model ensemble**；V11.2 原本的 OOF 實驗是 **trim after single-model prediction**。技術上 mask 操作一致，但整體統計效果不一定一致。
- V11.2 Fold 1 / Fold 2 的 `summary.json` 中 `inference` 欄寫成 `fold-0` 是報告文字常數未更新；請依 `records`、該 fold 基準 PQ 與 `--fold` 輸入確認實際評估資料。
- 若你希望更保守，先在已確認未參與模型訓練的驗證場景對三模型最終結果做評估；**不要把三個 V4 模型對原 707 張訓練影像的 Ensemble PQ 當成無洩漏評估**，因為每張影像可能被其餘模型看過。
- package 的原始碼與 V11.2 規則已用合成 mask 做一致性檢查；尚未在你的 Test 影像上產出候選 CSV，必須在本機跑完全套 Audit。
