# model_eval.py
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib
import matplotlib.pyplot as plt

# ---- 1) Load dataset
df = pd.read_csv("student_scores.csv")
print("Total rows loaded:", len(df))
print(df.head(), "\n")

# ---- 2) Prepare features and target
X = df[["Hours"]]   # keep as DataFrame (2D) with column name
y = df["Score"]

# ---- 3) Train/test split (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
print(f"Train rows: {len(X_train)}, Test rows: {len(X_test)}\n")

# ---- 4) Train model
model = LinearRegression()
model.fit(X_train, y_train)

# ---- 5) Predict on test set
y_pred = model.predict(X_test)

# ---- 6) Show actual vs predicted (sorted by Hours for readability)
results = pd.DataFrame({
    "Hours": X_test["Hours"].values,
    "Actual": y_test.values,
    "Predicted": np.round(y_pred, 2)
})
results = results.sort_values(by="Hours").reset_index(drop=True)
print("Test set results (sorted by Hours):")
print(results.to_string(index=False))
print()

# ---- 7) Evaluation metrics
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"MAE  = {mae:.3f}")
print(f"MSE  = {mse:.3f}")
print(f"R2   = {r2:.3f}\n")

# ---- 8) Save the trained model
joblib.dump(model, "linear_regressor_joblib.pkl")
print("Saved model -> linear_regressor_joblib.pkl")

# ---- 9) Plots
# 9a: Scatter + regression line
plt.figure(figsize=(6,4))
plt.scatter(X["Hours"], y, label="Data")
# regression line across range
X_line = np.linspace(X["Hours"].min(), X["Hours"].max(), 100).reshape(-1,1)
y_line = model.predict(X_line)
plt.plot(X_line, y_line, linewidth=2, label="Regression line")
plt.xlabel("Hours")
plt.ylabel("Score")
plt.title("Hours vs Score (with regression line)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("hours_vs_score_line.png", dpi=150)
plt.close()
print("Saved plot -> hours_vs_score_line.png")

# 9b: Residuals plot
residuals = y_test.values - y_pred
plt.figure(figsize=(6,4))
plt.scatter(X_test["Hours"], residuals)
plt.hlines(0, xmin=X_test["Hours"].min(), xmax=X_test["Hours"].max(), colors="r", linestyles="dashed")
plt.xlabel("Hours (test)")
plt.ylabel("Residual (Actual - Predicted)")
plt.title("Residuals on Test Set")
plt.grid(True)
plt.tight_layout()
plt.savefig("residuals_plot.png", dpi=150)
plt.close()
print("Saved plot -> residuals_plot.png")
