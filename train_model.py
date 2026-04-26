"""
train_model.py
--------------
This script trains a Linear Regression model on the student dataset,
evaluates it, and saves the trained model to a .pkl file.

Run this ONCE before launching the Streamlit app.
Usage:
    python train_model.py
"""

import pandas as pd
import numpy as np
import pickle
import os
import json

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error

# -------------------------------------------------------
# 1. Load the dataset
# -------------------------------------------------------
print("=" * 45)
print("   Student Performance Model Trainer")
print("=" * 45)

DATA_FILE  = "student_data.csv"
MODEL_FILE = "model.pkl"
INFO_FILE  = "model_info.json"

if not os.path.exists(DATA_FILE):
    print(f"\n[ERROR] '{DATA_FILE}' not found.")
    print("Make sure student_data.csv is in the same folder.")
    exit(1)

df = pd.read_csv(DATA_FILE)
print(f"\n[1] Dataset loaded  →  {df.shape[0]} rows, {df.shape[1]} columns")

# -------------------------------------------------------
# 2. Clean the data
# -------------------------------------------------------
df["StudyHours"] = df["StudyHours"].fillna(df["StudyHours"].mean())
df["Attendance"] = df["Attendance"].fillna(df["Attendance"].mean())
df["Marks"]      = df["Marks"].fillna(df["Marks"].mean())
print("[2] Missing values handled  ✅")

# -------------------------------------------------------
# 3. Prepare features and target
# -------------------------------------------------------
X = df[["StudyHours"]]   # input  (we can add more columns here later)
y = df["Marks"]           # output

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
print(f"[3] Data split  →  Train: {len(X_train)} | Test: {len(X_test)}")

# -------------------------------------------------------
# 4. Train the model
# -------------------------------------------------------
model = LinearRegression()
model.fit(X_train, y_train)
print("[4] Model trained  ✅")
print(f"    Formula → Marks = {model.coef_[0]:.2f} × StudyHours + {model.intercept_:.2f}")

# -------------------------------------------------------
# 5. Evaluate the model
# -------------------------------------------------------
y_pred = model.predict(X_test)
r2   = r2_score(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
mae  = np.mean(np.abs(y_test.values - y_pred))

print("\n[5] Model Performance:")
print(f"    R² Score : {r2:.4f}  (1.0 is perfect)")
print(f"    RMSE     : {rmse:.2f} marks average error")
print(f"    MAE      : {mae:.2f} marks average error")

# -------------------------------------------------------
# 6. Save the model as model.pkl
# -------------------------------------------------------
with open(MODEL_FILE, "wb") as f:
    pickle.dump(model, f)
print(f"\n[6] Model saved  →  '{MODEL_FILE}'  ✅")

# -------------------------------------------------------
# 7. Save model info as model_info.json
#    (useful for the Streamlit app to display info)
# -------------------------------------------------------
info = {
    "model_type":   "Linear Regression",
    "feature":      "StudyHours",
    "target":       "Marks",
    "coefficient":  round(float(model.coef_[0]), 4),
    "intercept":    round(float(model.intercept_), 4),
    "r2_score":     round(r2, 4),
    "rmse":         round(rmse, 4),
    "mae":          round(mae, 4),
    "train_size":   len(X_train),
    "test_size":    len(X_test),
    "formula":      f"Marks = {model.coef_[0]:.2f} x StudyHours + {model.intercept_:.2f}"
}

with open(INFO_FILE, "w") as f:
    json.dump(info, f, indent=4)
print(f"[7] Model info saved  →  '{INFO_FILE}'  ✅")

# -------------------------------------------------------
# 8. Quick prediction test
# -------------------------------------------------------
print("\n[8] Quick Prediction Check:")
test_hours = [1, 3, 5, 7, 9]
for h in test_hours:
    pred = max(0, min(100, round(model.predict([[h]])[0], 1)))
    print(f"    {h} hrs study  →  {pred} marks")

print("\n" + "=" * 45)
print("  All done! Now run:  streamlit run app.py")
print("=" * 45)
