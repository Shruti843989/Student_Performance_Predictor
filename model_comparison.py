"""
model_comparison.py
--------------------
This script compares three ML models on the student dataset:
  1. Linear Regression
  2. Decision Tree
  3. Random Forest

It shows which model predicts marks most accurately and
saves a comparison chart as 'model_comparison.png'.

Run with:
    python model_comparison.py
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle
import json

from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

# -------------------------------------------------------
# 1. Load and clean data
# -------------------------------------------------------
print("=" * 52)
print("   Model Comparison: LR vs DT vs RF")
print("=" * 52)

df = pd.read_csv("student_data.csv")
df["StudyHours"] = df["StudyHours"].fillna(df["StudyHours"].mean())
df["Attendance"] = df["Attendance"].fillna(df["Attendance"].mean())
df["Marks"]      = df["Marks"].fillna(df["Marks"].mean())

print(f"\n[1] Dataset loaded  →  {len(df)} students")

# -------------------------------------------------------
# 2. Features — using BOTH StudyHours and Attendance
#    (better than just one column!)
# -------------------------------------------------------
X = df[["StudyHours", "Attendance"]]
y = df["Marks"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
print(f"[2] Features used   →  StudyHours, Attendance")
print(f"    Train: {len(X_train)} | Test: {len(X_test)}")

# -------------------------------------------------------
# 3. Define all three models
# -------------------------------------------------------
models = {
    "Linear Regression": LinearRegression(),
    "Decision Tree":     DecisionTreeRegressor(max_depth=5, random_state=42),
    "Random Forest":     RandomForestRegressor(n_estimators=100, random_state=42),
}

# -------------------------------------------------------
# 4. Train, evaluate, collect results
# -------------------------------------------------------
print("\n[3] Training and Evaluating Models...\n")
print(f"{'Model':<22} {'R² Score':>10} {'RMSE':>8} {'MAE':>8} {'CV Score':>10}")
print("-" * 62)

results      = {}
predictions  = {}
best_model   = None
best_r2      = -999

for name, model in models.items():
    # Train
    model.fit(X_train, y_train)

    # Predict on test set
    y_pred = model.predict(X_test)
    y_pred = np.clip(y_pred, 0, 100)

    # Metrics
    r2   = r2_score(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae  = mean_absolute_error(y_test, y_pred)

    # Cross-validation (5-fold) — more reliable score
    cv_scores = cross_val_score(model, X, y, cv=5, scoring="r2")
    cv_mean   = cv_scores.mean()

    results[name] = {
        "r2":       round(r2, 4),
        "rmse":     round(rmse, 4),
        "mae":      round(mae, 4),
        "cv_score": round(cv_mean, 4),
    }
    predictions[name] = y_pred

    print(f"{name:<22} {r2:>10.4f} {rmse:>8.4f} {mae:>8.4f} {cv_mean:>10.4f}")

    if r2 > best_r2:
        best_r2    = r2
        best_model = name
        best_obj   = model

print("-" * 62)
print(f"\n🏆 Best Model: {best_model}  (R² = {best_r2:.4f})")

# -------------------------------------------------------
# 5. Save the best model as best_model.pkl
# -------------------------------------------------------
with open("best_model.pkl", "wb") as f:
    pickle.dump(best_obj, f)
print(f"\n[4] Best model saved  →  'best_model.pkl'  ✅")

# Save comparison results as JSON
with open("comparison_results.json", "w") as f:
    json.dump({
        "best_model": best_model,
        "results": results,
        "features": ["StudyHours", "Attendance"]
    }, f, indent=4)
print(f"[5] Results saved     →  'comparison_results.json'  ✅")

# -------------------------------------------------------
# 6. Charts — 4 subplots
# -------------------------------------------------------
print("\n[6] Creating comparison charts...")

fig, axes = plt.subplots(2, 2, figsize=(13, 10))
fig.suptitle("Model Comparison: Linear Regression vs Decision Tree vs Random Forest",
             fontsize=14, fontweight="bold", y=1.01)

model_names  = list(results.keys())
short_names  = ["Lin. Reg.", "Dec. Tree", "Ran. Forest"]
colors_bar   = ["#4285f4", "#fbbc04", "#34a853"]
colors_model = {"Linear Regression": "#4285f4",
                "Decision Tree":     "#fbbc04",
                "Random Forest":     "#34a853"}

# --- Chart 1: R² Score comparison ---
ax1 = axes[0, 0]
r2_vals = [results[m]["r2"] for m in model_names]
bars = ax1.bar(short_names, r2_vals, color=colors_bar, edgecolor="white", width=0.5)
for bar, val in zip(bars, r2_vals):
    ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.002,
             f"{val:.4f}", ha="center", fontsize=10, fontweight="bold")
ax1.set_ylim(0.90, 1.01)
ax1.set_title("R² Score (higher = better)")
ax1.set_ylabel("R² Score")
ax1.grid(axis="y", linestyle="--", alpha=0.4)

# --- Chart 2: RMSE comparison ---
ax2 = axes[0, 1]
rmse_vals = [results[m]["rmse"] for m in model_names]
bars2 = ax2.bar(short_names, rmse_vals, color=colors_bar, edgecolor="white", width=0.5)
for bar, val in zip(bars2, rmse_vals):
    ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
             f"{val:.2f}", ha="center", fontsize=10, fontweight="bold")
ax2.set_title("RMSE — Avg Error in Marks (lower = better)")
ax2.set_ylabel("RMSE (marks)")
ax2.grid(axis="y", linestyle="--", alpha=0.4)

# --- Chart 3: Actual vs Predicted for each model ---
ax3 = axes[1, 0]
actual = y_test.values
x_pos  = np.arange(len(actual))
ax3.plot(x_pos, actual, color="black", linewidth=1.5, label="Actual Marks", marker="o", markersize=4)
for name, color in colors_model.items():
    ax3.plot(x_pos, predictions[name], linestyle="--", color=color,
             linewidth=1.2, label=name, alpha=0.8)
ax3.set_title("Actual vs Predicted Marks (Test Set)")
ax3.set_xlabel("Student Index")
ax3.set_ylabel("Marks")
ax3.legend(fontsize=8)
ax3.grid(True, linestyle="--", alpha=0.3)

# --- Chart 4: Cross-validation scores ---
ax4 = axes[1, 1]
cv_vals = [results[m]["cv_score"] for m in model_names]
bars4 = ax4.bar(short_names, cv_vals, color=colors_bar, edgecolor="white", width=0.5)
for bar, val in zip(bars4, cv_vals):
    ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.002,
             f"{val:.4f}", ha="center", fontsize=10, fontweight="bold")
ax4.set_ylim(0.85, 1.01)
ax4.set_title("Cross-Validation Score (5-Fold, higher = better)")
ax4.set_ylabel("CV R² Score")
ax4.grid(axis="y", linestyle="--", alpha=0.4)

plt.tight_layout()
plt.savefig("model_comparison.png", dpi=150, bbox_inches="tight")
plt.show()
print("[7] Chart saved  →  'model_comparison.png'  ✅")

# -------------------------------------------------------
# 7. Final summary table
# -------------------------------------------------------
print("\n" + "=" * 52)
print("   FINAL SUMMARY")
print("=" * 52)
for name in model_names:
    r  = results[name]
    tag = " 🏆 BEST" if name == best_model else ""
    print(f"\n  {name}{tag}")
    print(f"    R² Score  : {r['r2']}")
    print(f"    RMSE      : {r['rmse']} marks avg error")
    print(f"    MAE       : {r['mae']} marks avg error")
    print(f"    CV Score  : {r['cv_score']}")

print("\n" + "=" * 52)
print(f"  Winner → {best_model}")
print("  Files created:")
print("    - best_model.pkl")
print("    - comparison_results.json")
print("    - model_comparison.png")
print("=" * 52)