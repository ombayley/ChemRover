#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Author: O. Bayley
Description: Dispatcher that picks the SMILES encoding method named in the `data.encoding` config.
             To add an encoding: write a function that takes a dataframe with a lower-case 'smiles'
             column and returns it with 'smiles' replaced by numeric feature columns (see ecfp.py),
             then add a branch for its name below.
"""
import re
import pandas as pd
from .RDKit_fp import add_rdk_fingerprints
from .ecfp import add_morgan_fingerprints


def add_fingerprint(df, method: str = 'rdkit', thresh: float = 0.9, n_bits: int = 2048) -> pd.DataFrame:
    """
    Replaces the 'smiles' column of ``df`` with numeric features from the chosen encoding.
    Args:
        df: pd.DataFrame - dataframe with a lower-case 'smiles' column plus any other features/target
        method: str - encoding name:
                'rdkit'                    RDKit 2D descriptors
                'ecfp4', 'ecfp6', ...      Morgan fingerprint; the number is the diameter (radius = n / 2)
                'fcfp4', 'fcfp6', ...      Morgan fingerprint on pharmacophoric atom features
                append '_count' to either for count vectors, e.g. 'ecfp4_count'
        thresh: float - (0-1) minimum non-NaN coverage for a descriptor to be kept ('rdkit' only)
        n_bits: int - fingerprint length (ECFP/FCFP only)
    Returns:
        pd.DataFrame - the other input columns joined with the encoded feature columns
    Raises:
        ValueError: if ``method`` is not a known encoding
    """
    if method == 'rdkit':
        return add_rdk_fingerprints(df, thresh)

    morgan = re.fullmatch(r"(ecfp|fcfp)(\d+)(_count)?", method)
    if morgan:
        kind, diameter, count = morgan.groups()
        if int(diameter) % 2:
            raise ValueError(f"'{method}': the number is the diameter (2 x radius), so it must be even, e.g. ecfp4")
        return add_morgan_fingerprints(df, radius=int(diameter) // 2, n_bits=n_bits,
                                       counts=count is not None, features=kind == 'fcfp')

    if method == 'mordred':
        raise NotImplementedError("Mordred not implemented yet")
    raise ValueError(f"Unrecognised encoding '{method}' (use rdkit, ecfp4/ecfp6, fcfp4/fcfp6, optionally with _count)")
