#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Author: O. Bayley
Description: Multi-layer perceptron regressor (sklearn MLPRegressor) in the ChemRover ModelBase interface.
             Configured by config/model/mlp.yaml. The network shape is given as `n_layers` x `n_units`
             (equal-width hidden layers) instead of sklearn's `hidden_layer_sizes` tuple, so Optuna can
             search it. Neural nets need scaled inputs and targets, so the config enables both.
"""
from sklearn.neural_network import MLPRegressor

from .sklearn_base import SklearnModel


class MLP(SklearnModel):
    """MLP regressor. Constructor args are documented on SklearnModel."""
    ESTIMATOR = MLPRegressor

    @property
    def PARAM_LIMS(self):
        """Optuna search space: {param: (low, high[, 'log'])} or {param: [choices]}."""
        return {
            'n_layers': (1, 3),                        # number of hidden layers
            'n_units': (16, 256, 'log'),               # neurons per hidden layer
            'alpha': (1e-6, 1e-1, 'log'),              # L2 regularisation
            'learning_rate_init': (1e-4, 1e-2, 'log'),
        }

    def _estimator(self):
        """Builds `hidden_layer_sizes` from `n_layers` x `n_units`."""
        params = dict(self.params)
        n_layers, n_units = params.pop("n_layers", 2), params.pop("n_units", 64)
        return MLPRegressor(hidden_layer_sizes=(n_units,) * n_layers, **params)
