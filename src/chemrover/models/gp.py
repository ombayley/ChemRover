#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Author: O. Bayley
Description: Gaussian process regressor (sklearn GaussianProcessRegressor) in the ChemRover ModelBase
             interface. Configured by config/model/gp.yaml.
             The kernel is chosen by name in the config and built here as
                 amplitude * <kernel>(length_scale) + white noise
             Its hyper-parameters (amplitude, length scale, noise) are fitted by maximising the
             marginal likelihood during fit(), so there is no Optuna search (empty PARAM_LIMS).
             Compare kernels with e.g. `model.kernel=rbf`.
"""
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel, Matern, RationalQuadratic, WhiteKernel

from .sklearn_base import SklearnModel

# kernel name (config value) -> stationary kernel on the standardised features
KERNELS = {
    "rbf": lambda: RBF(length_scale=10.0, length_scale_bounds=(1e-1, 1e3)),
    "matern32": lambda: Matern(length_scale=10.0, length_scale_bounds=(1e-1, 1e3), nu=1.5),
    "matern52": lambda: Matern(length_scale=10.0, length_scale_bounds=(1e-1, 1e3), nu=2.5),
    "rq": lambda: RationalQuadratic(length_scale=10.0, length_scale_bounds=(1e-1, 1e3)),
}


class GP(SklearnModel):
    """Gaussian process regressor. Constructor args are documented on SklearnModel."""
    ESTIMATOR = GaussianProcessRegressor

    @property
    def PARAM_LIMS(self):
        """No Optuna search: the kernel hyper-parameters are fitted by marginal likelihood."""
        return {}

    def _estimator(self):
        """Builds the kernel object from its name in the config (`kernel: matern52`)."""
        params = dict(self.params)
        name = params.pop("kernel", "matern52")
        if name not in KERNELS:
            raise ValueError(f"Unknown GP kernel '{name}' (choose from {list(KERNELS)})")
        kernel = ConstantKernel(1.0, (1e-3, 1e3)) * KERNELS[name]() + WhiteKernel(0.1, (1e-5, 1e1))
        return GaussianProcessRegressor(kernel=kernel, **params)
