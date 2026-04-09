# v2_final_code.py
# Final Refined Code: PGA Tour Strokes Gained Prediction
# SPC4004 Assessment 2
#
# Dataset: pgaTourData.csv  (PGA Tour Stats 2010-2018)
# Task:    Regression -- predict 'Average SG Total' from player performance stats
#
# Usage:   Place pgaTourData.csv in the same folder as this script, then run:
#              python v2_final_code.py

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
import warnings
warnings.filterwarnings("ignore")

# 1. LOAD & INSPECT DATA
df = pd.read_csv("pgaTourData.csv")
print(f"Dataset shape (raw): {df.shape}")

feature_cols = [
    "Fairway Percentage",
    "Avg Distance",
    "gir",
    "Average Putts",
    "Average Scrambling"
]
target_col = "Average SG Total"

# 2. PREPROCESSING
df = df.dropna(subset=[target_col])
for col in feature_cols:
    df[col] = df[col].fillna(df[col].median())

print(f"Dataset shape after cleaning: {df.shape}")
print(f"Missing values after cleaning:\n{df[feature_cols + [target_col]].isnull().sum()}\n")

X = df[feature_cols]
y = df[target_col]

# 3. TRAIN / TEST SPLIT
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
print(f"Train size: {len(X_train)}  |  Test size: {len(X_test)}\n")

# 4. MODEL A - Linear Regression (baseline)
lr_pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("model",  LinearRegression())
])
lr_pipeline.fit(X_train, y_train)
lr_preds = lr_pipeline.predict(X_test)
lr_r2    = r2_score(y_test, lr_preds)
lr_rmse  = np.sqrt(mean_squared_error(y_test, lr_preds))
lr_cv    = cross_val_score(lr_pipeline, X, y, cv=5, scoring="r2")

print("-- Linear Regression --")
print(f"  R2   (test): {lr_r2:.4f}")
print(f"  RMSE (test): {lr_rmse:.4f} strokes")
print(f"  CV R2 (5-fold): {lr_cv.mean():.4f} +/- {lr_cv.std():.4f}\n")

# 5. MODEL B - Random Forest Regressor (improved)
rf_pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("model",  RandomForestRegressor(
        n_estimators=200, max_depth=8, random_state=42, n_jobs=-1
    ))
])
rf_pipeline.fit(X_train, y_train)
rf_preds = rf_pipeline.predict(X_test)
rf_r2    = r2_score(y_test, rf_preds)
rf_rmse  = np.sqrt(mean_squared_error(y_test, rf_preds))
rf_cv    = cross_val_score(rf_pipeline, X, y, cv=5, scoring="r2")

print("-- Random Forest Regressor --")
print(f"  R2   (test): {rf_r2:.4f}")
print(f"  RMSE (test): {rf_rmse:.4f} strokes")
print(f"  CV R2 (5-fold): {rf_cv.mean():.4f} +/- {rf_cv.std():.4f}\n")

# 6. FIGURES
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
fig.suptitle("PGA Tour - Strokes Gained Total Prediction (Real Dataset)",
             fontsize=13, fontweight="bold")

ax1 = axes[0]
ax1.scatter(y_test, rf_preds, alpha=0.65, color="steelblue", edgecolors="white", s=55)
lims = [min(y_test.min(), rf_preds.min()) - 0.05,
        max(y_test.max(), rf_preds.max()) + 0.05]
ax1.plot(lims, lims, "r--", lw=1.5, label="Perfect prediction")
ax1.set_xlabel("Actual Strokes Gained Total")
ax1.set_ylabel("Predicted Strokes Gained Total")
ax1.set_title("Figure 1: Predicted vs Actual\n(Random Forest)")
ax1.legend(fontsize=8)
ax1.text(0.05, 0.92, f"R2 = {rf_r2:.3f}", transform=ax1.transAxes,
         fontsize=9, color="darkred")

ax2 = axes[1]
importances = rf_pipeline.named_steps["model"].feature_importances_
sorted_idx  = np.argsort(importances)
bar_colors  = ["#1565C0" if i == sorted_idx[-1] else "#90CAF9"
               for i in range(len(feature_cols))]
ax2.barh([feature_cols[i] for i in sorted_idx],
         importances[sorted_idx],
         color=[bar_colors[i] for i in range(len(sorted_idx))],
         edgecolor="white")
ax2.set_xlabel("Feature Importance")
ax2.set_title("Figure 2: Feature Importances\n(Random Forest)")

ax3 = axes[2]
residuals = y_test.values - rf_preds
ax3.scatter(rf_preds, residuals, alpha=0.65, color="coral", edgecolors="white", s=55)
ax3.axhline(0, color="black", linewidth=1.2, linestyle="--")
ax3.set_xlabel("Predicted Strokes Gained Total")
ax3.set_ylabel("Residual")
ax3.set_title("Figure 3: Residual Plot\n(Random Forest)")

plt.tight_layout()
plt.savefig("figures.png", dpi=150, bbox_inches="tight")
plt.close()
print("Figures saved to figures.png")

print("\n-- Model Comparison --")
print(f"{'Metric':<30} {'Linear Reg':>12} {'Random Forest':>14}")
print("-" * 58)
print(f"{'R2 (test set)':<30} {lr_r2:>12.4f} {rf_r2:>14.4f}")
print(f"{'RMSE (strokes)':<30} {lr_rmse:>12.4f} {rf_rmse:>14.4f}")
print(f"{'CV R2 mean (5-fold)':<30} {lr_cv.mean():>12.4f} {rf_cv.mean():>14.4f}")
print(f"{'CV R2 std':<30} {lr_cv.std():>12.4f} {rf_cv.std():>14.4f}")
