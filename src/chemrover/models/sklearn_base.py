#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Author: O. Bayley
Description: Shared ModelBase implementation for scikit-learn regressors (RF, SVM, KNN, ...).
             Unlike XGBoost, most sklearn models cannot handle NaNs or pandas 'category' columns,
             so every model is wrapped in the same preprocessing pipeline:
                 numeric columns     -> median imputation (+ standard scaling if scale_features)
                 categorical columns -> one-hot encoding (unseen categories are ignored)
                 target              -> optional standard scaling (scale_target), e.g. for SVR
             A concrete model only needs to set ESTIMATOR and define PARAM_LIMS (see models/rf.py).
"""
import joblib
import logging
import numpy as np
from pathlib import Path
from typing import Optional
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor, make_column_selector
from sklearn.exceptions import NotFittedError
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .base import ModelBase
LOG = logging.getLogger(__name__)


class SklearnModel(ModelBase):
    """
    Base class for scikit-learn regressors.
    Args:
        name: str - identifier used for the saved model file and the Optuna study name
        load_from: str | Path | None - optional saved model (.joblib) to load, skipping training
        scale_features: bool - standardise numeric features (needed by distance/kernel models)
        scale_target: bool - standardise the target during fitting (predictions are returned in
                      original units); keeps e.g. SVR's epsilon/C ranges independent of the target's units
        **params: passed straight to the ESTIMATOR class
    """
    ESTIMATOR: type = None   # the sklearn regressor class, set by subclasses

    def __init__(
            self,
            name: Optional[str] = None,
            load_from: Optional[str | Path] = None,
            scale_features: bool = False,
            scale_target: bool = False,
            **params,
    ):
        self._name = name or type(self).__name__.lower()
        self.scale_features = scale_features
        self.scale_target = scale_target
        self.params = params
        self._trained = False
        self.model = self._build()
        if load_from is not None:
            self.load(load_from)

    def _estimator(self):
        """The unfitted regressor. Override when config params need translating first (see models/gp.py)."""
        return self.ESTIMATOR(**self.params)

    def _build(self) -> Pipeline:
        """Build the (unfitted) preprocessing + regressor pipeline from the current params."""
        numeric = [SimpleImputer(strategy="median")] + ([StandardScaler()] if self.scale_features else [])
        preprocess = ColumnTransformer([
            ("num", make_pipeline(*numeric), make_column_selector(dtype_exclude="category")),
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False),
             make_column_selector(dtype_include="category")),
        ])
        regressor = self._estimator()
        if self.scale_target:
            regressor = TransformedTargetRegressor(regressor=regressor, transformer=StandardScaler())
        return Pipeline([("preprocess", preprocess), ("regressor", regressor)])

    @property
    def name(self):
        """Identifier used for the saved model file and the Optuna study name."""
        return self._name

    @property
    def FIXED_PARAMS(self):
        """Every configured param that is not tuned (i.e. not in PARAM_LIMS)."""
        return {k: v for k, v in self.params.items() if k not in self.PARAM_LIMS}

    def fit(self, X, y, X_test=None, y_test=None):
        """
        Fits the preprocessing pipeline and regressor.
        Args:
            X: pd.DataFrame - training features (NaNs and category columns allowed)
            y: pd.Series - training target
            X_test, y_test: accepted for interface compatibility; unused (no early stopping)
        """
        self.model.fit(X, y)
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

    def clone(self, **params) -> "SklearnModel":
        """New unfitted model of the same class and settings, with ``params`` taking precedence."""
        return type(self)(name=self._name, scale_features=self.scale_features,
                          scale_target=self.scale_target, **(self.params | params))

    def load(self, path: str | Path):
        """
        Loads a fitted pipeline saved with ``save``.
        Args:
            path: str | Path - the saved model .joblib file
        Raises:
            FileNotFoundError: if ``path`` is not a file
        """
        path = Path(path)
        if not path.is_file():
            raise FileNotFoundError(f"No saved model at {path}")
        self.model = joblib.load(path)
        self._trained = True
        LOG.info(f"Loaded model from {path}")

    def save(self, path: str | Path) -> Path:
        """
        Saves the fitted pipeline (preprocessing included) as '<name>.joblib'.
        Args:
            path: str | Path - directory to save into (created if missing)
        Returns:
            Path - the file written
        """
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        file = path / f"{self.name}.joblib"
        joblib.dump(self.model, file)
        LOG.info(f"Saved model to {file}")
        return file

    def set_params(self, **params):
        """Update hyper-parameters and rebuild the pipeline; the model must be fitted again afterwards."""
        self.params.update(params)
        self.model = self._build()
        self._trained = False
