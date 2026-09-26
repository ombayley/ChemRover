#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Author: O. Bayley
Description: Support vector regressor (sklearn SVR) in the ChemRover ModelBase interface.
             Configured by config/model/svm.yaml. SVR is sensitive to scale, so the config enables
             both feature and target standardisation; `epsilon` is therefore in standardised target units.
"""
from sklearn.svm import SVR

from .sklearn_base import SklearnModel


class SVM(SklearnModel):
    """Support vector regressor. Constructor args are documented on SklearnModel."""
    ESTIMATOR = SVR

    @property
    def PARAM_LIMS(self):
        """Optuna search space: {param: (low, high[, 'log'])} or {param: [choices]}."""
        return {
            'C': (0.1, 100.0, 'log'),
            'epsilon': (0.01, 1.0, 'log'),
            'gamma': (1e-4, 1.0, 'log'),
        }
