#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Author: O. Bayley
Description: **Add Desc**.
"""
import pandas as pd
from typing import Literal
from .RDKit_fp import add_rdk_fingerprints

def add_fingerprint(df, method: Literal['rdkit', 'mordred'] = 'rdkit', thresh:float = 0.9) -> pd.DataFrame:
    match method:
        case 'rdkit':
            return add_rdk_fingerprints(df, thresh)
        case 'mordred':
            raise NotImplementedError("Mordred not implemented yet")
        case _:
            raise ValueError("Unrecognized method")