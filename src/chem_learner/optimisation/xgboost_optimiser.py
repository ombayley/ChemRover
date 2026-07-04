import numpy as np
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

from xgboost import XGBRegressor

from skopt import BayesSearchCV
from skopt.space import Integer, Real

# Data
X, y = load_diabetes(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

xgb = XGBRegressor(
    objective="reg:squarederror",
    random_state=42,
    tree_method="hist",
    n_jobs=1,                 # important if BayesSearchCV uses parallelism
)

search = BayesSearchCV(
    estimator=xgb,
    search_spaces={
        "n_estimators": Integer(300, 5000),
        "learning_rate": Real(1e-3, 3e-1, prior="log-uniform"),
        "max_depth": Integer(2, 8),
        "min_child_weight": Real(1.0, 30.0, prior="log-uniform"),
        "subsample": Real(0.5, 1.0),
        "colsample_bytree": Real(0.5, 1.0),
        "gamma": Real(1e-8, 10.0, prior="log-uniform"),
        "reg_alpha": Real(1e-8, 10.0, prior="log-uniform"),
        "reg_lambda": Real(1e-3, 100.0, prior="log-uniform"),
    },
    n_iter=60,
    cv=5,
    scoring="neg_root_mean_squared_error",
    random_state=42,
    n_jobs=-1,
    verbose=2
)

search.fit(X_train, y_train)

best = search.best_estimator_
pred = best.predict(X_test)

rmse = np.sqrt(mean_squared_error(y_test, pred))
r2 = r2_score(y_test, pred)

print("Best params:", search.best_params_)
print(f"Test RMSE: {rmse:.3f}, R2: {r2:.3f}")

if __name__ == "__main__":
    pass
