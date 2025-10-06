# 📈 Stock Market Forecasting — Predicting Next Close / Delta Close / Direction

## 🧭 1. Problem Definition

We aim to forecast **future price movements** using historical OHLCV data.

**Targets we can predict:**

* **Next Close:** `close_{t+1}`
* **Delta Close:** `close_{t+1} - close_t`
* **Direction:**  `1` if price goes up, `0` if it goes down

---

## 🧹 2. Data Preparation

1. Load historical OHLCV data:
   `timestamp, open, high, low, close, volume`
2. Sort by timestamp and set it as the index.
3. Handle missing or duplicate data.
4. Resample if necessary (e.g., hourly or daily).

```python
df = df.sort_values('timestamp')
df = df.set_index('timestamp')
df = df.resample('1D').ffill()  # Example: Daily frequency
```

---

## ⚙️ 3. Feature Engineering

We create predictive features from the price and volume data.

### 🔹 Basic Features

* Lagged prices: `close.shift(1)`, `close.shift(2)`, ...
* Returns: `(close / close.shift(1)) - 1`
* Rolling statistics: mean, std, min, max

### 🔹 Technical Indicators

* RSI, MACD, ADX
* Bollinger Bands (upper/lower width)
* Moving Averages (SMA, EMA)
* Volume indicators (OBV, Volume change %)

```python
df['return_1'] = df['close'].pct_change()
df['ma_10'] = df['close'].rolling(10).mean()
df['volatility'] = df['close'].rolling(10).std()
```

---

## 🎯 4. Define the Target Variable

### (a) **Next Close**

```python
df['target_close'] = df['close'].shift(-1)
```

### (b) **Delta Close**

```python
df['target_delta'] = df['close'].shift(-1) - df['close']
```

### (c) **Direction (Binary Classification)**

```python
df['target_dir'] = (df['close'].shift(-1) > df['close']).astype(int)
```

Drop the last row (target is NaN after shift).

---

## 🧪 5. Train-Test Split (Chronological)

Always split time series data **by time**, not randomly.

```python
train_size = int(len(df) * 0.8)
train = df.iloc[:train_size]
test = df.iloc[train_size:]
```

---

## 📊 6. Feature Scaling

Normalize input features for better model convergence.

```python
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_train = scaler.fit_transform(train[features])
X_test = scaler.transform(test[features])
```

---

## 🤖 7. Model Training

You can start with **XGBoost**, which works well for tabular time-series data.

### Regression Example (for Close / Delta Close)

```python
from xgboost import XGBRegressor

model = XGBRegressor(
    n_estimators=500, learning_rate=0.05, max_depth=6, subsample=0.8
)
model.fit(X_train, y_train)
preds = model.predict(X_test)
```

### Classification Example (for Direction)

```python
from xgboost import XGBClassifier

model = XGBClassifier(
    n_estimators=300, learning_rate=0.05, max_depth=5, subsample=0.8
)
model.fit(X_train, y_train)
probs = model.predict_proba(X_test)[:, 1]
preds = (probs > 0.5).astype(int)
```

---

## 📈 8. Evaluation Metrics

### Regression

* RMSE (Root Mean Squared Error)
* MAE (Mean Absolute Error)
* R² Score

```python
from sklearn.metrics import mean_squared_error, r2_score

rmse = mean_squared_error(y_test, preds, squared=False)
r2 = r2_score(y_test, preds)
```

### Classification

* Accuracy
* Precision / Recall / F1-score
* ROC-AUC
* Directional Accuracy (for trading relevance)

---

## 💹 9. Backtesting & Signal Simulation

Simulate how predictions would perform as trades.

Examples:

* **Buy** when `predicted_direction == 1`
* **Sell** when `predicted_direction == 0`
* Compute returns, PnL, and Sharpe ratio

```python
test['pred_dir'] = preds
test['strategy_ret'] = test['pred_dir'] * test['return_1']
cum_ret = (1 + test['strategy_ret']).cumprod()
```

---

## 🔁 10. Model Optimization

* Tune hyperparameters (Optuna, GridSearch)
* Try alternative targets (delta instead of absolute)
* Add new features (sentiment, indices, etc.)
* Use ensemble models or neural networks (LSTM/Transformer)

---

## 🚀 11. (Optional) Deployment Pipeline

For production or live trading:

1. Automate daily data fetch.
2. Update features and retrain periodically.
3. Generate daily/hourly forecasts.
4. Store model and predictions for analysis.

---

### ✅ Summary

| Step | Goal                                               |
| ---- | -------------------------------------------------- |
| 1    | Define what to predict (Close / Delta / Direction) |
| 2    | Clean & prepare OHLCV data                         |
| 3    | Engineer lag, rolling & indicator features         |
| 4    | Create target variable                             |
| 5    | Split chronologically (no shuffling)               |
| 6    | Scale features                                     |
| 7    | Train model (XGBoost / LSTM etc.)                  |
| 8    | Evaluate performance                               |
| 9    | Backtest trading logic                             |
| 10   | Optimize & tune model                              |
| 11   | Automate & deploy                                  |

---

