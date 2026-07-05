# Linear Regression From Scratch

## 1. 專案目標

使用 Auto MPG Dataset，
不依賴任何機器學習框架，
自行實作 Linear Regression 與 Gradient Descent，
理解模型如何透過梯度下降學習最佳參數。

---

## 2. 使用資料集

Dataset：Auto MPG

預測目標(Label)

- mpg

使用 Feature

- weight

資料筆數

- 原始：398
- 清理後：392

---

## 3. 資料前處理

### 缺失值

horsepower 有 6 筆缺失值。

由於比例很低，因此直接刪除缺失資料。

### Feature

本次僅使用 weight。

原因：

weight 與 mpg 呈現明顯負相關，
適合作為第一個 Linear Regression 的實驗。

---

## 4. 模型

Prediction: ŷ = wx + b

Loss Function: Mean Squared Error (MSE)

Optimizer: Gradient Descent

---

## 5. 超參數

Learning Rate: 0.01

Epoch: 10000

初始 w: 50

初始 b: 50

---

## 6. 實驗結果

Final Loss: 18.6766165974193

Final Weight: -6.487381740339943

Final Bias: 23.445918367347026

---

## 7. 結果分析

### Loss Curve

Loss 隨著 Epoch 持續下降，
代表 Gradient Descent 能夠逐步降低模型誤差。

### Regression Line

回歸線能夠大致貼近資料趨勢，
表示模型成功學習到 weight 與 mpg 的線性關係。

---

## 8. 學習心得

本次最大的收穫不是完成 Linear Regression，
而是真正理解 Gradient Descent 的運作方式。

以前只知道模型會更新權重，
現在知道每一次更新都是：

Predict
→ Loss
→ Gradient
→ Update

形成完整的訓練流程。

另外，也理解到資料前處理的重要性，
如果直接使用含有缺失值的資料，
模型將無法正常訓練。

### 遇到的困難
超參數w、b的初始值只要改變，Final loss、Final w、Final b都會跟著改變
但是初始值不管設成甚麼，只要epoch夠多，learning rate 合理，都應該要收斂成同樣結果
後來發現是x的尺度太大，導致learning rate要設成0.0000001才不會爆炸，這就導致b更新速度超慢，幾乎沒什麼被改變
既然b幾乎不變，那訓練結果就會跟b的初始值有很大關係
解決方法是將x做標準化，使w和b的gradient的數量級差不多
未來如果遇到多features的模型要訓練，可以先對每個features做標準化，才不會遇到尺度相差太多的問題
