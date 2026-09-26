#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Author: O. Bayley
Description: Abstract Base Class (ABC) for the ML models to be used in this repo.
             Every model is instantiated by Hydra from its `model` config and must support the
             interface below so the trainer and CLI can treat all models the same way.
"""
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional

class ModelBase(ABC):
    """
    Common interface for all ChemRover models.

    To add a model:
        1. Subclass ModelBase and implement the abstract members below (see models/xgb.py).
        2. Add config/model/<name>.yaml with `_target_: chemrover.models.<module>.<Class>`
           plus the constructor arguments.
        3. Select it with `uv run chemrover model=<name>`.
    """

    def __getitem__(self, X):
        """Shorthand for ``predict(X)``."""
        return self.predict(X)

    @property
    @abstractmethod
    def PARAM_LIMS(self) -> dict:
        """Search space for tuning: {param: (low, high[, 'log'])} or {param: [choices]}. int bounds -> int param."""
        raise NotImplementedError

    @property
    @abstractmethod
    def FIXED_PARAMS(self) -> dict:
        """Params that are never tuned but must be set on every trial."""
        raise NotImplementedError

    @property
    @abstractmethod
    def name(self) -> str:
        """Identifier used for saved files and Optuna study names."""
        raise NotImplementedError

    @property
    def is_fitted(self) -> bool:
        """True once the model has been fitted or loaded from disk."""
        return getattr(self, "_trained", False)

    @property
    def best_iteration(self) -> Optional[int]:
        """Best boosting round from early stopping (None for models without early stopping)."""
        return None

    @abstractmethod
    def fit(self, X, y, X_test=None, y_test=None):
        """Fit on (X, y). Models that support early stopping use the tuning split (X_test, y_test) when given."""
        raise NotImplementedError

    @abstractmethod
    def predict(self, X):
        """Predict targets for X. Raises sklearn's NotFittedError if the model is not fitted."""
        raise NotImplementedError

    @abstractmethod
    def score(self, y_test, y_pred) -> dict:
        """Regression metrics comparing y_pred to y_test, as {"rmse", "mae", "r2"}."""
        raise NotImplementedError

    @abstractmethod
    def clone(self, **params) -> "ModelBase":
        """Return a new, unfitted copy of this model with ``params`` overriding the current ones."""
        raise NotImplementedError

    @abstractmethod
    def load(self, path: str | Path):
        """Load a fitted model from the file at ``path``."""
        raise NotImplementedError

    @abstractmethod
    def save(self, path: str | Path) -> Path:
        """Save the fitted model into the directory ``path`` and return the file written."""
        raise NotImplementedError

    @abstractmethod
    def set_params(self, **params):
        """Update hyper-parameters in place (does not refit)."""
        raise NotImplementedError
