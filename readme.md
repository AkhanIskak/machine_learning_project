[1 tool called]

# Complete Beginner's Guide to This Machine Learning Code

Let me explain everything from absolute zero. I'll go cell by cell and explain every term and concept.

---

## 🎯 What is Machine Learning?

**Machine Learning (ML)** is teaching computers to make predictions by showing them examples. Instead of writing rules like "if rating > 3, then high", we show the computer thousands of examples and it figures out the patterns itself.

**Analogy:** Imagine teaching a child to recognize cats. You don't give rules like "has whiskers, has tail, etc." - you just show them 1000 pictures of cats and they learn to recognize cats. That's ML!

---

## 📊 What This Project Does

**Goal:** Predict whether a user will give a movie a **high rating** (4 or 5 stars) or **low rating** (1, 2, or 3 stars).

This is called **Binary Classification** - sorting things into two categories (binary = two).

---

## Cell 0: Introduction

```markdown
# Audience Decode – ML Classification
```

This just describes the project. Key terms:
- **Classification** = sorting data into categories (high rating vs low rating)
- **Training/Validation/Test** = three separate portions of data (explained later)

---

## Cell 1: Importing Libraries

```python
import sqlite3  # connect to SQLite database
import numpy as np  # numerical computations
import pandas as pd  # data manipulation
```

### Key Libraries Explained:

| Library | What It Does |
|---------|-------------|
| `sqlite3` | Reads data from database files (.db) |
| `numpy` | Math operations on numbers |
| `pandas` | Works with tables of data (like Excel) |
| `sklearn` | The main ML library (scikit-learn) |
| `matplotlib` | Creates charts/graphs |

### sklearn components:

- **`ColumnTransformer`** - Applies different transformations to different columns
- **`Pipeline`** - Chains multiple steps together (like an assembly line)
- **`StandardScaler`** - Makes numbers comparable by adjusting their scale
- **`OneHotEncoder`** - Converts categories to numbers
- **`SimpleImputer`** - Fills in missing values
- **`LogisticRegression`, `RandomForestClassifier`, `GradientBoostingClassifier`** - Three different ML algorithms (explained below)

---

## Cell 2: Loading Data

```python
train_ratings, train_user_stats, train_movie_stats = load_data_from_db(TRAIN_DB_PATH)
```

**Output:**
```
Training data:
  viewer_ratings: (2817500, 5)  ← 2.8 million rows, 5 columns
  user_statistics: (405158, 10) ← 405K users, 10 columns
  movie_statistics: (12980, 11) ← ~13K movies, 11 columns
```

### 🔑 What is Train/Validation/Test Split?

Imagine you're studying for an exam:
- **Training data (70%)** = Your textbook - you study from this
- **Validation data (15%)** = Practice tests - you check your progress
- **Test data (15%)** = Final exam - you only look at this ONCE at the end

**Why split?** We need to test if the model works on data it has never seen. If we test on the same data we trained on, we're just checking if it memorized the answers, not if it actually learned.

---

## Cell 3: Preprocessing Data

```python
df["label_high"] = (df["rating"] >= 4).astype(int)  # 1 if rating >= 4, else 0
```

**What this does:**
1. Removes invalid ratings (not 1-5)
2. Removes rows with weird dates
3. Creates the **target variable** `label_high`:
   - `1` = High rating (4 or 5 stars)
   - `0` = Low rating (1, 2, or 3 stars)

**Output - Label Distribution:**
```
label_high
1    0.572452  ← 57% high ratings
0    0.427548  ← 43% low ratings
```

This is the **target** (what we're trying to predict).

---

## Cells 4-6: Feature Engineering

### What are Features?

**Features** = Information we give the model to make predictions. Think of them as clues.

**Example:** To predict if someone will like a movie:
- User's average rating (do they rate high or low in general?)
- Movie's average rating (is this movie generally liked?)
- How old is the movie?
- What day of the week was it rated?

```python
numeric_features = [
    "user_total_ratings",      # How many movies has user rated?
    "user_unique_movies",      # How many different movies?
    "user_avg_rating",         # User's typical rating
    "user_std_rating",         # How much do user's ratings vary?
    "movie_avg_rating",        # Movie's typical rating
    "movie_age_at_rating",     # How old was movie when rated?
    # ... etc
]
```

**Output after merging:**
```
Train merged shape: (2534660, 24)  ← 2.5 million records, 24 features
```

---

## Cell 7: Preparing X and y

```python
X_train, y_train = prepare_X_y(...)
```

### 🔑 Key Terminology:

- **X** = Features (input data) - all the clues we give the model
- **y** = Target (output) - what we want to predict (0 or 1)

**Output:**
```
X_train shape: (2534660, 14)  ← 2.5 million samples, 14 features each
y_train shape: (2534660,)     ← 2.5 million labels (0 or 1)
```

---

## Cell 8: Dataset Summary

```
Train size: 2,534,660 (70.0%)  ← Model learns from this
Val size: 542,941 (15.0%)      ← Model tuning
Test size: 543,038 (15.0%)     ← Final evaluation
```

---

## Cell 9: Data Preprocessing Pipeline

### What is Preprocessing?

Raw data often has problems:
- Missing values
- Numbers on different scales (age: 1-100, income: 10000-1000000)
- Text categories that need to be converted to numbers

### Numeric Features Pipeline:

```python
numeric_transformer = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),  # Fill missing with median
    ("scaler", StandardScaler()),                   # Normalize scale
])
```

**SimpleImputer** - Fills missing values. `strategy="median"` means use the middle value.

**StandardScaler** - Converts all numbers to the same scale.
- Before: Height in cm (150-200), Weight in kg (40-150), Age (1-100)
- After: All centered around 0, ranging roughly -3 to +3

**Why scale?** Some algorithms get confused when features have vastly different ranges. It's like comparing apples to skyscrapers.

### Categorical Features Pipeline:

```python
categorical_transformer = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(handle_unknown="ignore")),
])
```

**OneHotEncoder** - Converts categories to numbers.

Example - converting day of week:
```
Before: "Monday", "Tuesday", "Wednesday"

After (one-hot encoded):
         Mon  Tue  Wed
Monday:  [1,   0,   0]
Tuesday: [0,   1,   0]
Wednesday: [0, 0,   1]
```

---

## Cell 10: Defining the Models

### 🔑 The Three ML Algorithms Explained:

### 1. Logistic Regression

**What it is:** The simplest classification algorithm. Draws a line (or plane) to separate the two classes.

**How it works:**
1. Multiply each feature by a weight
2. Add them all up
3. Apply a special function (sigmoid) to convert to probability (0-1)
4. If probability > 0.5 → predict class 1, else class 0

**Visual:**
```
Imagine a 2D plot with points (class 0 as ○, class 1 as ●)

        ●  ●
      ● ●  ●
    ---------------  ← Logistic regression finds this line
    ○    ○
  ○   ○  ○
```

**Parameters:**
- `max_iter=1000` - Maximum number of learning cycles
- `class_weight="balanced"` - Give equal importance to both classes even if one has more samples

---

### 2. Random Forest

**What it is:** A "forest" of many decision trees that vote together.

**What is a Decision Tree?**
A tree of yes/no questions:
```
                Is movie_avg_rating > 3.5?
                    /           \
                  YES            NO
                  /               \
       Is user_avg_rating > 3?   Predict: LOW
            /         \
          YES          NO
          /             \
    Predict: HIGH   Predict: LOW
```

**How Random Forest works:**
1. Create 100 different decision trees
2. Each tree sees slightly different data (random sampling)
3. Each tree makes a prediction
4. **Final prediction = Majority vote** (like 100 experts voting)

**Why multiple trees?** One tree might make mistakes, but 100 trees voting together are much more accurate. It's like "wisdom of the crowd."

**Parameters:**
- `n_estimators=100` - Number of trees in the forest
- `random_state=42` - Seed for reproducibility (same random numbers each time)
- `n_jobs=-1` - Use all CPU cores (faster)

---

### 3. Gradient Boosting

**What it is:** Trees that learn from each other's mistakes.

**How it works:**
1. Train Tree 1 → makes some mistakes
2. Train Tree 2 → focuses on fixing Tree 1's mistakes
3. Train Tree 3 → focuses on fixing Tree 1+2's remaining mistakes
4. ... continue until very accurate

**Analogy:** 
- Student 1 takes the test, gets 70%
- Student 2 studies ONLY the questions Student 1 got wrong
- Student 3 studies ONLY the questions both got wrong
- Combined, they answer almost everything correctly

**Difference from Random Forest:**
- Random Forest: Trees work independently, then vote
- Gradient Boosting: Trees work sequentially, each fixing previous mistakes

---

## Cell 11: Training and Validation Results

```python
pipe.fit(X_train, y_train)  # TRAINING - model learns patterns
y_val_pred = pipe.predict(X_val)  # PREDICTION - model makes predictions
```

### 🔑 Understanding the Metrics:

Let's say we predict whether 100 movies get high ratings:

**Confusion Matrix** (what actually happened):
```
                    Predicted
                  LOW    HIGH
Actual  LOW      40       10    (50 actual low)
       HIGH      15       35    (50 actual high)
```

- **True Negative (TN) = 40**: Predicted LOW, actually LOW ✓
- **False Positive (FP) = 10**: Predicted HIGH, actually LOW ✗
- **True Positive (TP) = 35**: Predicted HIGH, actually HIGH ✓
- **False Negative (FN) = 15**: Predicted LOW, actually HIGH ✗

---

### Metrics Formulas and Meaning:

#### 1. **Accuracy** = (TP + TN) / Total
```
= (35 + 40) / 100 = 75%
```
"What percentage did we get right overall?"

---

#### 2. **Precision** = TP / (TP + FP)
```
= 35 / (35 + 10) = 77.8%
```
"When we predicted HIGH, how often were we correct?"

**Use when:** False positives are costly (e.g., spam filter - don't want important emails marked as spam)

---

#### 3. **Recall** = TP / (TP + FN)
```
= 35 / (35 + 15) = 70%
```
"Of all actual HIGHs, how many did we catch?"

**Use when:** False negatives are costly (e.g., cancer detection - don't want to miss actual cancer)

---

#### 4. **F1-Score** = 2 × (Precision × Recall) / (Precision + Recall)
```
= 2 × (0.778 × 0.70) / (0.778 + 0.70) = 73.7%
```
"Balance between Precision and Recall" - harmonic mean

---

#### 5. **ROC-AUC** (Area Under the ROC Curve)

ROC-AUC measures how well the model ranks predictions. 
- **1.0** = Perfect (always ranks positive higher than negative)
- **0.5** = Random guessing (no better than flipping a coin)
- **0.0** = Perfectly wrong

**In this project:**
- Logistic Regression: 0.8849
- Random Forest: 0.9130
- **Gradient Boosting: 0.9153** (best!)

---

### Actual Results from Validation:

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
|-------|----------|-----------|--------|-----|---------|
| Logistic Regression | 80.3% | 84.7% | 80.1% | 82.3% | 88.5% |
| Random Forest | 81.7% | 83.1% | 85.4% | 84.2% | 91.3% |
| **Gradient Boosting** | **81.9%** | 82.6% | **86.7%** | **84.6%** | **91.5%** |

**Winner: Gradient Boosting** (highest F1-score of 84.6%)

---

## Cell 12: Final Test Results

```python
best_pipe.fit(X_train_full, y_train_full)  # Retrain on train+val combined
y_test_pred = best_pipe.predict(X_test)    # Final prediction on test set
```

### Why retrain on train+val?

- During development: Use validation to pick the best model
- For final model: Use ALL available training data (train + validation)
- Test only at the very end, ONCE

### Final Test Performance (Gradient Boosting):

```
Test Accuracy: 0.8181   ← 81.8% correct overall
Test Precision: 0.8299  ← When we say "high rating", we're right 83% of time
Test Recall: 0.8589     ← We catch 86% of all actual high ratings
Test F1: 0.8441         ← Balanced score of 84.4%
Test ROC-AUC: 0.9147    ← Excellent ranking ability (0.91 out of 1.0)
```

### Final Confusion Matrix:

```
Confusion matrix (test):
[[176676  54856]   ← Out of 231,532 actual LOW: 176,676 correct, 54,856 wrong
 [ 43949 267557]]  ← Out of 311,506 actual HIGH: 267,557 correct, 43,949 wrong
```

**Visual interpretation:**
```
                        Predicted
                    LOW         HIGH
Actual  LOW      176,676 ✓    54,856 ✗  
       HIGH       43,949 ✗   267,557 ✓
```

---

## 📈 Summary: What Did We Learn?

1. **Best Model:** Gradient Boosting
2. **Accuracy:** ~82% (correctly predicts high/low rating 82% of the time)
3. **Key Insight:** The model is slightly better at catching high ratings (86% recall) than low ratings (76% recall)

---

## 🎓 Quick Glossary

| Term | Simple Explanation |
|------|-------------------|
| **Classification** | Sorting data into categories |
| **Binary** | Two options (yes/no, high/low) |
| **Feature** | Input variable (clue for prediction) |
| **Target/Label** | What we're trying to predict |
| **Training** | Model learns patterns from data |
| **Overfitting** | Model memorizes training data, fails on new data |
| **Accuracy** | % of correct predictions |
| **Precision** | When predicting positive, how often correct? |
| **Recall** | Of all actual positives, how many caught? |
| **F1** | Balance of precision and recall |
| **ROC-AUC** | How well model ranks predictions (1.0 = perfect) |
| **Pipeline** | Chain of processing steps |
| **Imputer** | Fills missing values |
| **Scaler** | Normalizes number ranges |
| **OneHot** | Converts categories to numbers |

Do you want me to explain any specific part in more detail?