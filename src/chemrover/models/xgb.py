#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Author: O. Bayley
Description: **Add Desc**.
"""
import numpy as np
from pathlib import Path
from typing import Optional
from xgboost import XGBRegressor
from sklearn.exceptions import NotFittedError
from sklearn.metrics import mean_squared_error, r2_score

from .base import ModelBase
from chemrover.logger import get_logger
LOG = get_logger(name="xgb_log")

class XGB(ModelBase):
    def __init__(self, **params):
        self.params = params
        self._trained = False
        self.model = XGBRegressor()
        self.load()

    @property
    def name(self):
        if "name" in self.params and self.params["name"] is not None:
            return self.params["name"]
        return "Unnamed XGB"

    @property
    def PARAM_LIMS(self):
        return {
            'max_depth': (2, 10),
            'learning_rate': (1e-2, 0.3),
            'n_estimators': (100, 3000),
            'subsample': (0.5, 1.0),
            'colsample_bytree': (0.5, 1.0),
            'min_child_weight': (1, 10),
            'gamma': (0.0, 5.0),
        }

    @property
    def FIXED_PARAMS(self):
        return {
            "enable_categorical": self.params["enable_categorical"] if "enable_categorical" in self.params else True,
            "tree_method": self.params["enable_categorical"] if "enable_categorical" in self.params else "hist",
            'early_stopping_rounds': self.params["enable_categorical"] if "enable_categorical" in self.params else 50,
            'n_jobs': self.params["enable_categorical"] if "enable_categorical" in self.params else 1,
        }

    def fit(self, X, y):
        self.model.fit(X, y)
        self._trained = True

    def predict(self, X):
        if not self._trained:
            raise NotFittedError("This model has not been fitted yet.")
        return self.model.predict(X)

    def score(self, y_test, y_pred):
        return {
            "rmse": np.sqrt(mean_squared_error(y_test, y_pred)),
            "mae": np.mean(np.abs(y_test - y_pred)),
            "r2": r2_score(y_test, y_pred)
        }

    def load(self, path: Optional[str | Path] = None):
        if path is None and "path" in self.params and self.params["path"] is not None:
            path = self.params['path']

        path = Path(path) / f"{self.name}.json"
        if path.exists() and path.is_file():
            self.model.load_model(path)
            self._trained = True
            LOG.info(f"Loaded model from {path}")
        else:
            self.model.set_params(**self.params)
            LOG.info(f"Loaded model from known params. NO model found at {path}")


    def save(self, name: str, path: Optional[str | Path] = None):
        if path is None and "path" in self.params and self.params["path"] is not None:
            path = self.params['path']
        path = Path(path)
        if path.is_file():
            path = path.parent
        file = path / f"{name}.json"
        self.model.save_model(file)

    def set_params(self, **params):
        self.model.set_params(**params)

    @property
    def best_iteration(self):
        return self.model.best_iteration