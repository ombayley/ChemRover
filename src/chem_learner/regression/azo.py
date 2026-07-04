#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Author: O. Bayley
Description: **Add Desc**.
"""
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from src.chem_learner.descriptors.RDKit_Descriptor import load_azo_dataset
from src.chem_learner.descriptors.feature_correlation import get_top_features

# Data
X, y = load_azo_dataset()
corr_df, X = get_top_features(
    x=X,
    y=y,
    min_correlation=0.0,
    method="spearman",
    return_filtered_x=True
)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

models = {
    # "RandomForestOpt": RandomForestRegressor(max_depth=25, max_features='sqrt', min_samples_leaf=7, min_samples_split=2, n_estimators=771),
    "XGBoostOpt": XGBRegressor(
        n_estimators=3507, learning_rate=0.00148, max_depth=7,
        subsample=0.616, colsample_bytree=0.5, gamma=0.832, min_child_weight=30.0,
        reg_alpha=3.026e-06, reg_lambda=0.001
    ),
}

for name, model in models.items():
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, preds))
    r2 = r2_score(y_test, preds)
    plt.plot(y_test, preds, 'o')
    mae = np.mean(np.abs(y_test - preds))
    print(f"{name}: RMSE={rmse:.3f}, MAE={mae}, R2={r2:.3f}")

    plt.show()

if __name__ == "__main__":
    pass
