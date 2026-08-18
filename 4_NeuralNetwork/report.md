# Neural Network 實作報告

## 1. 學習目標

從零實作簡單的 Neural Network，理解神經元、Activation Function、Forward Propagation、Backpropagation 與 Gradient Descent，並使用真實資料集進行二元分類。

## 2. 實作內容

- 實作 Sigmoid、ReLU 與 ReLU Derivative
- 使用矩陣運算建立 Hidden Layer
- 實作 Forward Propagation
- 使用 Binary Cross Entropy 計算 Loss
- 推導並實作 Backpropagation
- 使用 Gradient Descent 更新 Weight 與 Bias
- 封裝 NeuralNetwork Class
- 使用 Breast Cancer Wisconsin Dataset 進行訓練與測試
- 比較不同 Hidden Size 對模型表現的影響

## 3. Neural Network 結構

本次使用一層 Hidden Layer：

Input Layer → Hidden Layer (ReLU) → Output Layer (Sigmoid)

Forward Propagation：

Z1 = XW1 + b1
A1 = ReLU(Z1)

Z2 = A1W2 + b2
A2 = Sigmoid(Z2)

Output Layer 使用 Sigmoid 產生二元分類的預測機率。

## 4. Backpropagation

透過 Chain Rule 將 Loss 的誤差由 Output Layer 傳回 Hidden Layer：

- `dZ2 = A2 - y`
- `dW2 = (A1.T @ dZ2) / m`
- `db2 = mean(dZ2)`

dA1 = dZ2 @ W2.T
dZ1 = dA1 * ReLU'(Z1)

dW1 = (X.T @ dZ1) / m
db1 = mean(dZ1)

最後使用 Gradient Descent 更新 W1、b1、W2、b2。

## 5. Hidden Size 實驗

固定 Learning Rate = 0.01、Epochs = 5000，並固定 Random Seed，比較不同 Hidden Size。

| Hidden Size | Train Accuracy | Test Accuracy | Final Loss |
|-------------|----------------|---------------|------------|
| 2 | 0.9890 | 0.9649 | 0.079525 |
| 4 | 0.9890 | 0.9649 | 0.043982 |
| 6 | 0.9890 | 0.9737 | 0.043500 |
| 8 | 0.9890 | 0.9737 | 0.043332 |
| 10 | 0.9890 | 0.9737 | 0.043871 |

## 6. 實驗結果

所有 Hidden Size 的 Train Accuracy 都為 0.9890。

Hidden Size 從 2 增加至 4 時，Final Loss 明顯下降；Hidden Size 為 6、8、10 時，Test Accuracy 提升至 0.9737，但三者之間沒有明顯差異。

實驗結果顯示，增加 Hidden Neurons 可以提高模型的表達能力，但當模型容量已經足夠後，繼續增加 Neurons 不一定能持續改善模型表現。

## 7. 學習心得

透過從零實作 Neural Network，更清楚理解 Forward Propagation 與 Backpropagation 的運作方式。Neural Network 的訓練方式與先前實作的 Logistic Regression 有許多相似之處，但加入 Hidden Layer 後，可以學習更複雜的特徵。

也學到了以下重要概念：

- Activation Function 的重要作用之一是為神經網路引入非線性。如果沒有 Activation Function，即使堆疊多個 Linear Layer，最後仍然可以合併成一個線性模型，無法學習複雜的非線性關係。

- 一個 Linear Layer 的 Weight Matrix 形狀為：

  W.shape = (input_size, output_size)

  其中 input_size 代表這一層接收到的特徵數量，output_size 代表這一層的神經元數量。例如從 30 個輸入特徵連接到 8 個 Hidden Neurons，則 W.shape = (30, 8)。

另外，透過 Hidden Size 實驗了解到神經元數量屬於 Hyperparameter，並不是越多越好，需要透過實驗比較模型的訓練與測試結果。