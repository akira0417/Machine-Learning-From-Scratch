# CNN 學習報告

## 1. CNN 基本原理

本週學習 CNN（Convolutional Neural Network）的基本架構與運作方式。

主要學習內容：
- CNN 適合處理具有空間結構的影像資料。
- Convolution 使用 Kernel 在影像上滑動以擷取局部特徵，並產生 Feature Map。
- Padding 可以控制 Convolution 後 Feature Map 的尺寸。
- Stride 決定 Kernel 每次移動的距離。
- Pooling 可以縮小 Feature Map，降低後續計算量。
- CNN 中的 Kernel Weight 可以透過 Backpropagation 與 Gradient Descent 自動學習。
- 多層 Convolution 可以逐漸學習由簡單到複雜的影像特徵。

## 2. PyTorch 與 CNN 實作

使用 PyTorch 建立 CNN，學習 Tensor、Dataset、DataLoader 與 Mini-batch Training。

CNN 基本流程：

Input → Convolution → ReLU → Max Pooling → Flatten → Fully Connected

使用 MNIST 手寫數字資料集進行 0～9 的多類別影像分類，並使用 CrossEntropyLoss 作為 Loss Function。

訓練流程：

Forward → Loss → Backward → Optimizer → Update Parameters

## 3. CNN 架構實驗

比較使用 1 層與 2 層 Convolution 的模型。

| CNN 架構 | Final Loss | Train Accuracy | Test Accuracy |
| --- | ---: | ---: | ---: |
| 1 Conv | 0.2085 | 94.05% | 94.70% |
| 2 Conv | 0.0960 | 97.08% | 97.18% |

增加第二層 Convolution 後，Test Accuracy 從 94.70% 提升至 97.18%。多層 Convolution 能逐步學習較複雜的影像特徵，因此提升了分類效果。

## 4. Learning Rate 實驗

固定 2 Conv、Batch Size = 32、Epoch = 5，比較不同 Learning Rate。

| Learning Rate | Final Loss | Train Accuracy | Test Accuracy |
| ---: | ---: | ---: | ---: |
| 0.001 | 0.3381 | 90.06% | 91.34% |
| 0.01 | 0.0960 | 97.08% | 97.18% |
| 0.1 | 0.0394 | 98.76% | 98.62% |

Learning Rate = 0.001 時收斂速度較慢；Learning Rate = 0.1 在本次 5 Epoch 的實驗中收斂最快，並得到最高的最終 Test Accuracy。

## 5. Batch Size 實驗

固定 2 Conv、Learning Rate = 0.1、Epoch = 5，比較不同 Batch Size。

| Batch Size | Final Loss | Train Accuracy | Test Accuracy |
| ---: | ---: | ---: | ---: |
| 16 | 0.0397 | 98.75% | 98.13% |
| 32 | 0.0394 | 98.76% | 98.62% |
| 64 | 0.0549 | 98.30% | 98.15% |

三種 Batch Size 都能取得良好的分類效果，其中 Batch Size = 32 在本次實驗中的最終 Test Accuracy 最高。

Batch Size 較小時，每個 Epoch 會進行較多次參數更新；Batch Size 較大時，每個 Epoch 的更新次數較少。

## 6. 學習心得

透過本週的學習，我了解了 Convolution、Padding、Stride、Pooling 等 CNN 基本原理，也從 NumPy 的簡單實作進一步使用 PyTorch 建立並訓練完整 CNN。

實驗中也觀察到 CNN 層數、Learning Rate、Batch Size 都會影響模型的訓練結果與收斂速度。最終使用 2 層 Convolution、Learning Rate = 0.1、Batch Size = 32，在 MNIST 上得到 98.62% 的最終 Test Accuracy。