#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Author: O. Bayley
Description: ChemRover - ML predictors for azobenzene photoswitch properties.
             Importing the package registers the ``project_root`` OmegaConf resolver, which the
             configs use (``${project_root:}``) so that no absolute paths are hard-coded.
"""
from pathlib import Path
from omegaconf import OmegaConf

PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = PROJECT_ROOT / "outputs"
CONFIG_DIR = PROJECT_ROOT / "config"

if not OmegaConf.has_resolver("project_root"):
    OmegaConf.register_new_resolver("project_root", lambda: PROJECT_ROOT.as_posix())
