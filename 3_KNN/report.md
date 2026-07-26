# KNN 分類實作

## 專案目的

本專案使用 Python 從零開始實作 K-Nearest Neighbors（KNN）演算法，不使用任何機器學習套件，了解 KNN 的分類流程，並使用 Breast Cancer Wisconsin Dataset 進行分類與模型評估。

## 使用資料集

- Breast Cancer Wisconsin Dataset
- 共 569 筆資料
- 30 個特徵
- 分類為良性（Benign）與惡性（Malignant）

## 實作內容

本次實作包含：

- 歐氏距離（Euclidean Distance）
- KNN 模型
- fit()
- predict()
- Accuracy
- Precision
- Recall
- F1-score
- Confusion Matrix

## K 值實驗結果

本次固定使用相同的資料集與資料前處理方式，分別測試 K = 1、3、5、7、9，比較不同 K 值對模型分類效果的影響。評估指標包含 Accuracy、Precision、Recall 與 F1-score。

| K | Accuracy | Precision | Recall | F1-score |
|---|:--------:|:---------:|:------:|:--------:|
| 1 | 0.94 | 0.95 | 0.91 | 0.93 |
| 3 | 0.94 | 0.97 | 0.89 | 0.93 |
| 5 | 0.92 | 0.97 | 0.85 | 0.90 |
| 7 | 0.93 | 0.97 | 0.87 | 0.92 |
| 9 | 0.93 | 0.97 | 0.87 | 0.92 |

## 實驗分析

由實驗結果可觀察到，不同的 K 值會影響 KNN 的分類表現。

當 K = 1 時，Accuracy、Recall 與 F1-score 均為本次實驗中最佳，表示模型能較好地辨識正類樣本，但也較容易受到個別資料點的影響。

當 K 增加至 3 時，Precision 提升至 0.97，代表模型在預測為正類時具有較高的準確性，但 Recall 略為下降，表示漏判正類的情況稍微增加。

當 K 持續增加至 5、7、9 時，Accuracy、Recall 與 F1-score 均略有下降，顯示 K 值過大會使模型在分類時受到較多鄰居影響，造成決策邊界較平滑，降低對部分樣本的辨識能力。

綜合本次實驗結果，K = 1 與 K = 3 均具有較佳的整體表現。其中 K = 1 在 Accuracy、Recall 與 F1-score 上最佳，而 K = 3 則具有最高的 Precision。若以整體分類能力作為考量，本次資料集以 K = 1 的表現最佳。

## 學習心得

透過本專案，我了解了 KNN 的原理與實作方式，並學會使用歐氏距離找出最近鄰居，再利用多數決完成分類。同時也了解不同 K 值會影響模型的預測結果，並學會使用 Accuracy、Precision、Recall 與 F1-score 評估模型表現。完成本專案後，對 KNN 演算法及分類模型有更深入的理解。