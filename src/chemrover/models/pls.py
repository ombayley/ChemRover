#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Author: O. Bayley
Description: Partial least squares regressor (sklearn PLSRegression) in the ChemRover ModelBase
             interface. Configured by config/model/pls.yaml. The standard linear baseline in
             chemometrics: projects the (correlated) descriptors onto a few latent components that
             best explain the target, then fits a linear model on those.
"""
from sklearn.cross_decomposition import PLSRegression

from .sklearn_base import SklearnModel


class PLS(SklearnModel):
    """PLS regressor. Constructor args are documented on SklearnModel."""
    ESTIMATOR = PLSRegression

    @property
    def PARAM_LIMS(self):
        """Optuna search space: {param: (low, high[, 'log'])} or {param: [choices]}."""
        return {
            'n_components': (1, 30),   # number of latent components
        }
