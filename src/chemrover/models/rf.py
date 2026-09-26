#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Author: O. Bayley
Description: Random forest regressor (sklearn RandomForestRegressor) in the ChemRover ModelBase interface.
             Configured by config/model/rf.yaml. Preprocessing (imputation, one-hot solvent) comes
             from SklearnModel; trees don't need feature scaling.
"""
from sklearn.ensemble import RandomForestRegressor

from .sklearn_base import SklearnModel


class RF(SklearnModel):
    """Random forest regressor. Constructor args are documented on SklearnModel."""
    ESTIMATOR = RandomForestRegressor

    @property
    def PARAM_LIMS(self):
        """Optuna search space: {param: (low, high[, 'log'])} or {param: [choices]}."""
        return {
            'n_estimators': (100, 1000),
            'max_depth': (2, 30),
            'min_samples_split': (2, 20),
            'min_samples_leaf': (1, 10),
            'max_features': (0.1, 1.0),   # fraction of features considered at each split
        }
