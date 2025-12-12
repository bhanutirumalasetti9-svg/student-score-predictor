# cv_compare.py
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import cross_val_score, KFold
from sklearn.pipeline import make_pipeline
from sklearn.metrics import make_scorer, mean_absolute_error, mean_squared_error, r2_score

df = pd.read_csv("student_scores.csv")
X = df[["Hours"]].values
y = df["Score"].values

# scorers (negated where sklearn expects higher=better)
def mae_scorer(y_true, y_pred): return mean_absolute_error(y_true, y_pred)
def mse_scorer(y_true, y_pred): return mean_squared_error(y_true, y_pred)
def r2_scorer(y_true, y_pred): return r2_score(y_true, y_pred)

scorers = {
    "MAE": make_scorer(mae_scorer, greater_is_better=False),
    "MSE": make_scorer(mse_scorer, greater_is_better=False),
    "R2": make_scorer(r2_scorer)
}

kf = KFold(n_splits=5, shuffle=True, random_state=42)

models = {
    "Linear": make_pipeline(LinearRegression()),
    "Poly2": make_pipeline(PolynomialFeatures(degree=2, include_bias=False), LinearRegression()),
    "Ridge_alpha1": make_pipeline(Ridge(alpha=1.0))
}

for name, model in models.items():
    print(f"\nModel: {name}")
    # MAE (we invert sign because greater_is_better=False)
    mae_scores = -cross_val_score(model, X, y, cv=kf, scoring=scorers["MAE"])
    mse_scores = -cross_val_score(model, X, y, cv=kf, scoring=scorers["MSE"])
    r2_scores = cross_val_score(model, X, y, cv=kf, scoring=scorers["R2"])
    print(f" MAE mean ± std: {mae_scores.mean():.3f} ± {mae_scores.std():.3f}")
    print(f" MSE mean ± std: {mse_scores.mean():.3f} ± {mse_scores.std():.3f}")
    print(f" R2  mean ± std: {r2_scores.mean():.3f} ± {r2_scores.std():.3f}")
