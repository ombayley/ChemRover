#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Author: O. Bayley
Description: XGBoost regressor wrapped in the ChemRover ModelBase interface.
             Configured by config/model/xgb.yaml. Handles pandas 'category' columns (e.g. solvent)
             natively via `enable_categorical`, and uses early stopping when the tuning (test) split is given.
"""
import logging
import numpy as np
from pathlib import Path
from typing import Optional
from xgboost import XGBRegressor
from sklearn.exceptions import NotFittedError
from sklearn.metrics import mean_squared_error, r2_score

from .base import ModelBase
LOG = logging.getLogger(__name__)

class XGB(ModelBase):
    """
    XGBoost regressor.
    Args:
        name: str - identifier used for the saved model file and the Optuna study name
        load_from: str | Path | None - optional saved model (.json) to load, skipping training
        **params: passed straight to xgboost.XGBRegressor
    """
    def __init__(self, name: str = "xgb", load_from: Optional[str | Path] = None, **params):
        self._name = name
        self.params = params
        self._trained = False
        self.model = XGBRegressor(**params)
        if load_from is not None:
            self.load(load_from)

    @property
    def name(self):
        """Identifier used for the saved model file and the Optuna study name."""
        return self._name

    @property
    def PARAM_LIMS(self):
        """Optuna search space: {param: (low, high)}; int bounds are sampled as ints."""
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
        """Non-tuned params taken from the config (with defaults), re-applied on every trial."""
        return {
            "enable_categorical": self.params.get("enable_categorical", True),
            "tree_method": self.params.get("tree_method", "hist"),
            "early_stopping_rounds": self.params.get("early_stopping_rounds", 50),
            "n_jobs": self.params.get("n_jobs", 1),
        }

    def fit(self, X, y, X_test=None, y_test=None):
        """
        Fits the regressor.
        Args:
            X: pd.DataFrame - training features (category columns allowed)
            y: pd.Series - training target
            X_test, y_test: optional tuning split. If given, it is used for early stopping
                          (`early_stopping_rounds`); otherwise early stopping is disabled for this fit.
        """
        if X_test is not None and y_test is not None:
            self.model.fit(X, y, eval_set=[(X_test, y_test)], verbose=False)
        else:
            # early stopping needs an evaluation set, so switch it off for this fit
            self.model.set_params(early_stopping_rounds=None)
            self.model.fit(X, y, verbose=False)
            self.model.set_params(early_stopping_rounds=self.params.get("early_stopping_rounds"))
        self._trained = True

    def predict(self, X):
        """Predict targets for X (same columns/dtypes as the training features)."""
        if not self._trained:
            raise NotFittedError("This model has not been fitted yet.")
        return self.model.predict(X)

    def score(self, y_test, y_pred):
        """Returns {"rmse", "mae", "r2"} as plain floats."""
        return {
            "rmse": float(np.sqrt(mean_squared_error(y_test, y_pred))),
            "mae": float(np.mean(np.abs(y_test - y_pred))),
            "r2": float(r2_score(y_test, y_pred)),
        }

    def clone(self, **params) -> "XGB":
        """New unfitted XGB with the same name and params, with ``params`` taking precedence."""
        return XGB(name=self._name, **(self.params | params))

    def load(self, path: str | Path):
        """
        Loads a fitted booster saved with ``save``.
        Args:
            path: str | Path - the saved model .json file
        Raises:
            FileNotFoundError: if ``path`` is not a file
        """
        path = Path(path)
        if not path.is_file():
            raise FileNotFoundError(f"No saved model at {path}")
        self.model.load_model(path)
        self._trained = True
        LOG.info(f"Loaded model from {path}")

    def save(self, path: str | Path) -> Path:
        """
        Saves the fitted booster as '<name>.json'.
        Args:
            path: str | Path - directory to save into (created if missing)
        Returns:
            Path - the file written
        """
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        file = path / f"{self.name}.json"
        self.model.save_model(file)
        LOG.info(f"Saved model to {file}")
        return file

    def set_params(self, **params):
        """Update hyper-parameters on both the wrapper and the underlying XGBRegressor."""
        self.params.update(params)
        self.model.set_params(**params)

    @property
    def best_iteration(self) -> Optional[int]:
        """Best boosting round from the last early-stopped fit, else None."""
        try:
            return self.model.best_iteration
        except AttributeError:  # only defined when early stopping was used
            return None
