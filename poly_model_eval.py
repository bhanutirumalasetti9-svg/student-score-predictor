# poly_model_eval.py
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib
import matplotlib.pyplot as plt

df = pd.read_csv("student_scores.csv")
X = df[["Hours"]]
y = df["Score"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# polynomial transform (degree 2)
poly = PolynomialFeatures(degree=2, include_bias=False)
X_train_p = poly.fit_transform(X_train)
X_test_p  = poly.transform(X_test)

model = LinearRegression()
model.fit(X_train_p, y_train)

y_pred = model.predict(X_test_p)

results = pd.DataFrame({
    "Hours": X_test["Hours"].values,
    "Actual": y_test.values,
    "Predicted": np.round(y_pred, 2)
}).sort_values("Hours").reset_index(drop=True)
print("Test set results (poly deg2):")
print(results.to_string(index=False))
print()

mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print(f"MAE  = {mae:.3f}")
print(f"MSE  = {mse:.3f}")
print(f"R2   = {r2:.3f}")

joblib.dump((poly, model), "poly_degree2_model.pkl")
print("Saved -> poly_degree2_model.pkl")

# Plot data and polynomial curve
X_line = np.linspace(X["Hours"].min(), X["Hours"].max(), 200).reshape(-1,1)
X_line_p = poly.transform(X_line)
y_line = model.predict(X_line_p)

plt.figure(figsize=(6,4))
plt.scatter(X["Hours"], y, label="Data")
plt.plot(X_line, y_line, linewidth=2, label="Poly degree=2 fit")
plt.xlabel("Hours"); plt.ylabel("Score")
plt.title("Polynomial fit (degree=2)")
plt.legend(); plt.grid(True); plt.tight_layout()
plt.savefig("poly_deg2_fit.png", dpi=150)
plt.close()
print("Saved plot -> poly_deg2_fit.png")
