#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Author: O. Bayley
Description: k-nearest-neighbours regressor (sklearn KNeighborsRegressor) in the ChemRover ModelBase
             interface. Configured by config/model/knn.yaml. Distances need comparable feature
             scales, so the config enables feature standardisation.
"""
from sklearn.neighbors import KNeighborsRegressor

from .sklearn_base import SklearnModel


class KNN(SklearnModel):
    """k-nearest-neighbours regressor. Constructor args are documented on SklearnModel."""
    ESTIMATOR = KNeighborsRegressor

    @property
    def PARAM_LIMS(self):
        """Optuna search space: {param: (low, high[, 'log'])} or {param: [choices]}."""
        return {
            'n_neighbors': (1, 30),
            'weights': ['uniform', 'distance'],
            'p': [1, 2],                  # 1 = Manhattan, 2 = Euclidean distance
        }
