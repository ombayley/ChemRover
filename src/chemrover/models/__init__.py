#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Author: O. Bayley
Description: ML models sharing the ModelBase interface. Concrete models are instantiated by
             Hydra from the `model` config group via their `_target_`.
"""
from .base import ModelBase