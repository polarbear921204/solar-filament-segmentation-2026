# V11.2：V4 單模型的局部 mask 修正受控實驗

**目的：** 先根據 V11.1 的 9 個 IoU 0.45～0.50 失敗案例測試「保守的一像素影像引導修正」是否在 Fold 0 的**全部 365 筆、236 張實際影像**改善 PQ。不要用這 9 筆 GT 在 Test 中挑選該擴張／縮小哪條 mask。

**重要：** 此程式從你的 `solar_filament_v4/src` 匯入原始 `predict_native_image`、`panoptic_quality`、COCO annotation 解析，不修改舊版 V4 主程式或 Kaggle 0.38 CSV。每個 record 推論一次，再以同一批 prediction 比較四個版本：

1. `baseline`：不修改 V4 mask。
2. `grow1`：只在原 mask 之外一個像素、且局部比周圍更暗的位置增加像素（影像 8-bit 亮度差 >1.5；每 mask 面積增長不超過 30%）。
3. `trim1`：只從原 mask 邊界拿掉明顯亮於周圍的像素（差 <-1.5；每 mask 刪除不超過 15%）。
4. `grow_trim1`：先修剪再擴張，同樣遵守上述上限。

這些只是**預先固定的測試規則**，不保證比 V4 更好，不是經過 GT 學出來的 mask 生成模型。兩種方法都無法恢復完全不存在的 prediction，這 8 個 no-overlap GT 要留待另行處理。

## 用法

在你伺服器原始專案根目錄執行：

```bash
cd /home/ivan/SSD/solar_filament
unzip -o v11_2_mask_refinement.zip

# 可選：先跑 3 筆 smoke test（不做正式 PQ 選擇）
python v11_2_mask_refinement/scripts/evaluate_refinement_oof.py \
  --model runs/segment/runs_v4/yolov8m_patch_v4_fold0/weights/best.pt \
  --manifest v8_fold0_manifest.csv \
  --baseline-csv v8_fold0_baseline.csv \
  --limit-records 3 --device 0 --output-dir v11_2_smoke

# 正式跑完整 Fold 0。只做一次 YOLO 推論，同時比較 4 組。
python v11_2_mask_refinement/scripts/evaluate_refinement_oof.py \
  --model runs/segment/runs_v4/yolov8m_patch_v4_fold0/weights/best.pt \
  --manifest v8_fold0_manifest.csv \
  --baseline-csv v8_fold0_baseline.csv \
  --device 0 --output-dir v11_2_results
```

若 ZIP 解壓到巢狀兩層目錄，使用 `find . -name evaluate_refinement_oof.py` 查看實際路徑。

正式運行時會先逐筆核對 baseline `tp/fp/fn/sum_iou` 是否與既有 V4 Fold 0 原版完全一致（小數容差 0.00003）。**若失敗，必須停止，不要拿 challengers 的結果選參數。**

## 結果與下一步

產生：

- `v11_2_results/summary.json`：四組完整 Native PQ、TP/FP/FN、相對 baseline 的差異、physical-image cluster bootstrap 95% CI。
- `v11_2_results/record_metrics.csv`：逐筆紀錄結果。

只有 Fold 0 全體有正向趨勢、FP 受控，才**固定規則**到 Fold 1、2 驗證。不要只在 19 個挑出的困難 GT 上決定是否採用，也不要直接把結果當作三模型 Ensemble 的 PQ，因為 Fold 0 是 OOF 單模型而非 V4 Ensemble。

**可能的環境問題：** `solar_filament_v4/` 必須存在於專案目錄；原始 COCO JSON 和 image path 須在 `v8_fold0_manifest.csv` 中仍可用；需安裝原本 V4 使用的 `ultralytics`、`opencv-python`、`pycocotools`、`numpy`、`pandas`。

**評估條件：** 使用 1024 tile、768 stride、imgsz 1024、conf 0.30、max instances 15，以及原本 V4 的合併／去重設定。預測的 instance 順序與數量不變，調整後仍不允許重疊像素。GT 只在最後計算 PQ 才用。
