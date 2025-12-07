# Gradient Boosting Speed Comparison

## Why is Gradient Boosting Slow?

### Your Dataset
- **Training samples**: 2,534,660 rows (~2.5 million)
- **Features**: 14 (12 numeric + 2 categorical)
- **Classes**: 2 (binary classification: high rating vs low rating)

### The Problem: Sequential Training

| Model | Training Method | Can Parallelize? | Speed on Your Data |
|-------|----------------|------------------|-------------------|
| **Logistic Regression** | Gradient descent on linear model | Partially | ~30 seconds ⚡ |
| **Random Forest** | Builds 100 trees independently | ✅ Yes (`n_jobs=-1`) | ~5-10 minutes ⚡⚡ |
| **GradientBoosting** | Builds 100 trees **sequentially** | ❌ No | ~30-60 minutes 🐌 |
| **HistGradientBoosting** | Histogram-based, optimized | Partially | ~3-5 minutes ⚡⚡⚡ |

## Why GradientBoosting is Slow

### 1. **Sequential Nature**
```
Tree 1: Learn from data → residuals₁
Tree 2: Learn from residuals₁ → residuals₂  ← Must wait for Tree 1
Tree 3: Learn from residuals₂ → residuals₃  ← Must wait for Tree 2
...
Tree 100: Learn from residuals₉₉             ← Must wait for Tree 99
```

Each tree **must wait** for the previous tree to complete!

### 2. **Computational Complexity Per Tree**
For each of 100 trees:
- Compute gradients for **all 2.5M samples**
- For each feature, sort and find best split
- Update residuals for next iteration

**Total operations**: O(n_trees × n_samples × n_features × depth)

### 3. **No Parallelization**
- Random Forest: `n_jobs=-1` → uses all CPU cores → **10x faster**
- GradientBoosting: Single-threaded → **slow**

## Solution: Use HistGradientBoostingClassifier

### What's Different?

| Aspect | GradientBoosting | HistGradientBoosting |
|--------|------------------|---------------------|
| Algorithm | Exact split finding | Histogram binning |
| Memory | Stores all values | Bins values (e.g., 256 bins) |
| Split finding | O(n log n) sorting | O(n) binning |
| Missing values | Requires imputation | Native support |
| Speed on large data | Slow | **10-100x faster** |
| Accuracy | Slightly better | Nearly identical |

### Speed Comparison on Your Data (2.5M rows)

```
GradientBoostingClassifier:
  ████████████████████████████████████████  30-60 minutes

HistGradientBoostingClassifier:
  ███  3-5 minutes  ⚡⚡⚡ (10x faster!)
```

## Implementation (Already Updated in Your Notebook!)

```python
from sklearn.ensemble import HistGradientBoostingClassifier

hist_gb_clf = HistGradientBoostingClassifier(
    max_iter=100,          # equivalent to n_estimators
    learning_rate=0.1,
    max_depth=5,           # can go deeper since it's faster
    random_state=42,
    verbose=0,             # set to 1 to see training progress
)
```

## Even Faster Options (Optional)

If you need more speed, install external libraries:

### XGBoost
```bash
pip install xgboost
```

```python
import xgboost as xgb

xgb_clf = xgb.XGBClassifier(
    n_estimators=100,
    max_depth=5,
    learning_rate=0.1,
    n_jobs=-1,              # Can parallelize!
    random_state=42,
)
```

**Speed**: ~2-4 minutes (15x faster than standard GB)

### LightGBM
```bash
pip install lightgbm
```

```python
import lightgbm as lgb

lgb_clf = lgb.LGBMClassifier(
    n_estimators=100,
    max_depth=5,
    learning_rate=0.1,
    n_jobs=-1,
    random_state=42,
)
```

**Speed**: ~1-3 minutes (20-30x faster than standard GB)

## Summary

### The Change I Made:
✅ Replaced `GradientBoostingClassifier` → `HistGradientBoostingClassifier`

### Expected Results:
- **Training time**: 30-60 min → 3-5 min (10x faster)
- **Accuracy**: Nearly identical
- **No code changes needed**: Drop-in replacement in scikit-learn

### Why It Works:
- Uses histogram binning instead of exact splits
- More efficient memory access patterns
- Optimized C++ implementation
- Better cache utilization

## Your Updated Notebook

Cell 10 now uses:
```python
"Hist Gradient Boosting": Pipeline([
    ("preprocess", preprocessor),
    ("model", hist_gb_clf),  # Much faster!
])
```

The old slow version is commented out if you want to compare later.

---

**Bottom Line**: For datasets with 100K+ rows, always use `HistGradientBoostingClassifier` instead of `GradientBoostingClassifier`. It's a free 10-100x speedup with no accuracy loss! 🚀

