from skopt import BayesSearchCV
from skopt.space import Integer, Categorical
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
import numpy as np

X, y = load_diabetes(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

rf = RandomForestRegressor(random_state=42)

search = BayesSearchCV(
    rf,
    search_spaces={
        "n_estimators": Integer(100, 1200),
        "max_depth": Integer(2, 30),
        "min_samples_split": Integer(2, 20),
        "min_samples_leaf": Integer(1, 10),
        "max_features": Categorical(["sqrt", "log2", 1.0]),
        "criterion": Categorical(["squared_error", "absolute_error"]),
    },
    n_iter=30,
    cv=5,
    scoring="neg_root_mean_squared_error",  # <- regression scoring
    random_state=42,
    n_jobs=-1,
    verbose=2
)

search.fit(X_train, y_train)
best = search.best_estimator_
pred = best.predict(X_test)
rmse = np.sqrt(mean_squared_error(y_test, pred))
print("Best params:", search.best_params_)
print("Test RMSE:", rmse)

if __name__ == "__main__":
    pass