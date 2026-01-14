#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Author: O. Bayley
Description: **Add Desc**.
"""
import numpy as np
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score

from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor

# Data
X, y = load_diabetes(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

models = {
    "NN (MLPRegressor)": Pipeline([
        ("scaler", StandardScaler()),
        ("model", MLPRegressor(hidden_layer_sizes=(64, 64), max_iter=2000, random_state=42))
    ]),
    "RandomForest": RandomForestRegressor(n_estimators=400, random_state=42),
    "RandomForestOpt": RandomForestRegressor(max_depth=25, max_features='sqrt', min_samples_leaf=7, min_samples_split=2, n_estimators=771),
    "XGBoost": XGBRegressor(
        n_estimators=600, learning_rate=0.05, max_depth=4,
        subsample=0.8, colsample_bytree=0.8, random_state=42
    ),
}

for name, model in models.items():
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    # rmse = mean_squared_error(y_test, preds, squared=False)
    rmse = np.sqrt(mean_squared_error(y_test, preds))
    r2 = r2_score(y_test, preds)
    print(f"{name}: RMSE={rmse:.3f}, R2={r2:.3f}")

if __name__ == "__main__":
    pass
