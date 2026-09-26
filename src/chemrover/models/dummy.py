#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Author: O. Bayley
Description: Baseline regressor (sklearn DummyRegressor) in the ChemRover ModelBase interface.
             Configured by config/model/dummy.yaml. Ignores the features and always predicts the
             training mean (or median), so it sets the floor every real model must beat
             (R2 ~ 0 on the val set). Nothing to tune, so the trainer skips Optuna.
"""
from sklearn.dummy import DummyRegressor

from .sklearn_base import SklearnModel


class Dummy(SklearnModel):
    """Mean/median baseline. Constructor args are documented on SklearnModel."""
    ESTIMATOR = DummyRegressor

    @property
    def PARAM_LIMS(self):
        """No search space: the trainer fits once with the configured params."""
        return {}
