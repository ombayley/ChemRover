#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Author: O. Bayley
Description: Abstract Base Class (ABC) for the ML models to be used in this repo.
"""
from abc import ABC, abstractmethod

class ModelBase(ABC):

    def __getitem__(self, X):
        return self.predict(X)

    @property
    @abstractmethod
    def PARAM_LIMS(self):
        raise NotImplementedError

    @property
    @abstractmethod
    def FIXED_PARAMS(self):
        raise NotImplementedError

    @property
    @abstractmethod
    def name(self):
        raise NotImplementedError

    @abstractmethod
    def fit(self, X, y):
        raise NotImplementedError

    @abstractmethod
    def predict(self, X):
        raise NotImplementedError

    @abstractmethod
    def score(self, y_test, y_pred):
        raise NotImplementedError

    @abstractmethod
    def load(self, path):
        raise NotImplementedError

    @abstractmethod
    def save(self, name, path):
        raise NotImplementedError

    @abstractmethod
    def set_params(self, **params):
        raise NotImplementedError
