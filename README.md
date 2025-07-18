# 📈 Amazon Stock Time Series Forecasting using GRU, LSTM, and Hybrid Deep Networks

> Time-series forecasting on Amazon’s stock (AMZN) using GRU, LSTM, and a Hybrid CNN-GRU architecture. Trained on over 25 years of stock data to model market behavior and price patterns with high precision.

---

## 🚀 Project Highlights

* ⏳ **Data Range**: 1997–2025
* 🧮 **Models Used**: GRU, LSTM, Hybrid CNN-GRU
* 🎯 **Metrics**: MAE, RMSE, R², EVS
* 🧠 **Libraries**: TensorFlow, Scikit-learn, Pandas, Matplotlib, Seaborn

---

## 📂 Dataset Overview

**Source**: [Kaggle - Amazon Complete Stocks Data](https://www.kaggle.com/datasets/matiflatif/amazon-complete-stocks-data)
**File Used**: `AMZN_1997-03-16_2025-01-31.csv`
**Features**:

* `Date` (converted to `datetime`)
* `Open`, `High`, `Low`, `Close`, `Adj Close`, `Volume`

✅ No missing values
✅ Chronologically sorted
✅ Scaled using `MinMaxScaler`

---

## 📊 Exploratory Analysis

```python
# Line Plot: Close Price Over Time
plt.plot(data['Date'], data['Close'])

# Volume Trend
plt.plot(data['Date'], data['Volume'])

# Correlation Matrix
sns.heatmap(data.corr(), annot=True)
```

* 📈 Close price trends upward long-term
* 🔍 High correlation between `Adj Close` and `Close`

---

## 🛠️ Preprocessing Pipeline

* ✅ Sequence creation using 10-day window
* ✅ `train_test_split` (15% test, 10% validation from remaining)
* ✅ Feature-target split:
  **Features**: `['Open', 'High', 'Low', 'Volume', 'Adj Close']`
  **Target**: `Close`

---

## 🧠 Models

### 1. 🔄 GRU Model

```python
GRU(512) ➝ Dropout ➝ GRU(256) ➝ Dropout ➝ Dense(128) ➝ Dense(1)
```

* 🔧 `loss='mse'`, optimizer: Adam (lr=0.0001)
* 🧪 Early stopping + learning rate scheduler

**Results (Test Set):**

| Metric | Value  |
| ------ | ------ |
| MAE    | 1.13   |
| RMSE   | 2.44   |
| R²     | 0.9983 |

---

### 2. 🧬 LSTM Model

```python
LSTM(512) ➝ Dropout ➝ LSTM(256) ➝ Dropout ➝ Dense(128) ➝ Dense(1)
```

* LSTM handled temporal dependencies well
* Required longer training time than GRU

**Results (Test Set):**

| Metric | Value  |
| ------ | ------ |
| MAE    | 1.39   |
| RMSE   | 3.08   |
| R²     | 0.9973 |

---

### 3. 🧠 CNN-GRU Hybrid Model

```python
Conv1D ➝ MaxPool ➝ GRU(256) ➝ GRU(128) ➝ Dense(64) ➝ Dense(1)
```

* 👓 Extracted local temporal features
* ⚠️ Slight overfitting without aggressive dropout

| Metric | Value  |
| ------ | ------ |
| MAE    | 4.66   |
| RMSE   | 6.82   |
| R²     | 0.9869 |


## 🧪 Evaluation Summary


| Model     | MAE (Test) | RMSE (Test) | R² (Test) | MAE (Val) | RMSE (Val) | R² (Val) |
|-----------|------------|-------------|-----------|-----------|-------------|----------|
| **GRU**   | 1.13       | 2.44        | 0.9983    | 1.12      | 2.22        | 0.9985   |
| **LSTM**  | 1.39       | 3.08        | 0.9973    | 1.31      | 2.78        | 0.9976   |
| **CNN-GRU** | 4.66     | 6.82        | 0.9869    | 4.83      | 6.88        | 0.9855   |


---

## ✅ Conclusion

* GRU **outperformed** other models in both speed and accuracy.
* CNN-GRU added feature richness but struggled with generalization.
* LSTM was robust but slightly lagged behind GRU on metrics.

---

## 💡 Future Work

* 🔍 Integrate macroeconomic indicators (e.g., inflation, interest rates)
* 📦 Try Transformer-based models for longer temporal context
* 🎯 Add sentiment data from news headlines

---

## 📜 License

This repository is licensed under the **MIT License**.
