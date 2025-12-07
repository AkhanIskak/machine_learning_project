"""
Gradient Boosting Speed Optimization Examples
"""

# ============================================================================
# OPTION 1: Use HistGradientBoostingClassifier (MUCH FASTER - RECOMMENDED)
# ============================================================================
from sklearn.ensemble import HistGradientBoostingClassifier

# This is a modern, optimized version of gradient boosting
# - Uses histogram-based algorithm (like LightGBM/XGBoost)
# - Native support for missing values
# - Much faster on large datasets (10-100x speedup)
# - Can handle categorical features natively

hist_gb_clf = HistGradientBoostingClassifier(
    max_iter=100,           # equivalent to n_estimators
    learning_rate=0.1,
    max_depth=5,            # can go deeper since it's faster
    random_state=42,
    verbose=1,              # show progress
)

# ============================================================================
# OPTION 2: Optimize Standard GradientBoostingClassifier
# ============================================================================
from sklearn.ensemble import GradientBoostingClassifier

# Reduce number of trees and limit depth
fast_gb_clf = GradientBoostingClassifier(
    n_estimators=50,        # Reduce from 100 (default) to 50
    max_depth=3,            # Keep shallow (default is 3)
    learning_rate=0.1,      # Default
    subsample=0.8,          # Use 80% of samples per tree (speeds up)
    min_samples_split=200,  # Require more samples to split (speeds up)
    min_samples_leaf=100,   # Larger leaves = less splits
    random_state=42,
    verbose=1,              # Show progress
)

# ============================================================================
# OPTION 3: Use XGBoost or LightGBM (Fastest - External Libraries)
# ============================================================================

# XGBoost - Install with: pip install xgboost
try:
    import xgboost as xgb
    
    xgb_clf = xgb.XGBClassifier(
        n_estimators=100,
        max_depth=5,
        learning_rate=0.1,
        n_jobs=-1,              # CAN parallelize!
        random_state=42,
        verbosity=1,
    )
    print("✓ XGBoost available")
except ImportError:
    print("✗ XGBoost not installed. Install with: pip install xgboost")

# LightGBM - Install with: pip install lightgbm
try:
    import lightgbm as lgb
    
    lgb_clf = lgb.LGBMClassifier(
        n_estimators=100,
        max_depth=5,
        learning_rate=0.1,
        n_jobs=-1,              # CAN parallelize!
        random_state=42,
        verbose=1,
    )
    print("✓ LightGBM available")
except ImportError:
    print("✗ LightGBM not installed. Install with: pip install lightgbm")

# ============================================================================
# COMPARISON: Training Time Estimates on ~2.8M samples
# ============================================================================
# GradientBoostingClassifier (default):     ~30-60 minutes (SLOW)
# GradientBoostingClassifier (optimized):   ~15-20 minutes (MEDIUM)
# HistGradientBoostingClassifier:           ~3-5 minutes (FAST)
# XGBoost (with parallelization):           ~2-4 minutes (VERY FAST)
# LightGBM (with parallelization):          ~1-3 minutes (VERY FAST)
# ============================================================================

