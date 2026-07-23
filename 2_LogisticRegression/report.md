# Logistic Regression From Scratch

## 專案目的

本專案以 NumPy 從零實作 Logistic Regression，不使用 sklearn 的模型，了解二元分類模型的訓練流程，包含資料前處理、模型訓練、預測以及模型評估。

---

## 使用資料集

使用 Wisconsin Diagnostic Breast Cancer Dataset。

- M：惡性腫瘤（1）
- B：良性腫瘤（0）

共使用 30 個特徵進行分類。

---

## 實作內容

本專案完成以下功能：

- 自行實作 Train/Test Split
- 自行實作 Standardization
- 實作 Sigmoid Function
- 實作 Binary Cross Entropy Loss
- 實作 Gradient Descent
- 實作 Logistic Regression
- 實作 Accuracy、Precision、Recall、F1-score
- 實作 Confusion Matrix

---

## 模型結果

使用參數：

- Learning Rate：0.01
- Epoch：2000

結果：

| 指標 | 數值 |
|------|------|
| Train Accuracy | 0.9890 |
| Test Accuracy | 0.9649 |
| Final Loss | 0.077157 |

模型在測試集達到約 96% 的準確率，且訓練集與測試集的差距不大，沒有明顯的過度擬合。

---

## Learning Rate 實驗

本次將 Epoch 固定為 500，並比較以下 Learning Rate：

- 10
- 1
- 0.1
- 0.01
- 0.001

實驗結果顯示，Learning Rate 為 10 時得到最低的 Final Loss。

在固定 Epoch 的情況下，較小的 Learning Rate 每次參數更新幅度較小，因此經過 500 次訓練後可能仍未完全收斂。Learning Rate 為 10 時更新速度較快，且在本次資料集與標準化設定下仍能穩定訓練，因此得到較低的 Loss。

不過，這不代表 Learning Rate 越大越好。若 Learning Rate 過大，參數可能跨過最佳解，使 Loss 震盪或發散。因此，Learning Rate 的適合範圍會受到資料尺度、模型與 Epoch 數量影響，需要透過實驗判斷。

---

## Epoch 實驗

比較不同 Epoch 後發現：

- Epoch 太少時，模型尚未收斂。
- 隨著 Epoch 增加，Loss 持續下降，模型準確率提高。
- Epoch 足夠後，模型改善幅度逐漸變小。

本次實驗中，Epoch=2000 時已獲得穩定的分類效果。

---

## 其他實驗

將 Bias 初始值由 0 改為 100 後，模型仍能收斂，但需要更多 Epoch 才能達到相近的結果，說明初始化方式會影響模型收斂速度。

---

## 學習心得

透過自行實作 Logistic Regression，我更了解模型的完整流程，而不是只會呼叫現成函式。除了理解 Sigmoid、Binary Cross Entropy 與 Gradient Descent 的運作方式，也學會了資料前處理的重要性，以及 Learning Rate、Epoch 等超參數對模型訓練的影響。