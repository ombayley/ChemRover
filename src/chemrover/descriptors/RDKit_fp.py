#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Author: O. Bayley
Description: **Add Desc**.
"""
import rdkit
import pandas as pd
import numpy as np
from typing import Optional
from rdkit import Chem
from rdkit.Chem import Descriptors
from chemrover.logger import get_logger
LOG = get_logger(name="RDKit_descr")

def _to_mol(smiles: str) -> Optional[Chem.Mol]:
    """Convert SMILES to Mol Object"""
    return Chem.MolFromSmiles(smiles)


def _calc_desc(mol: Chem.Mol) -> dict:
    """Calculates the descriptors for the molecule as defined in desc_fns"""
    return Descriptors.CalcMolDescriptors(mol)

def _calc_desc_debug(mol: Chem.Mol) -> dict:
    """
    Alternative to _calc_desc that uses the same calculations but allows clearer debugging.
    Was needed to identify a Numpy-Rdkit install issue derived from conda vs pip installs in the env.
    Retained in case of future debug need.
    """
    desc_fns = Descriptors._descList
    val_dict = {}
    for name, fn in desc_fns:
        try:
            val_dict[name] = fn(mol)
            LOG.debug(f"Calculated value: {val_dict[name]} for param: {name}")
        except Exception:
            val_dict[name] = np.nan
    return val_dict


def _get_empty_cols(df: pd.DataFrame, lim: float) -> list[str]:
    """find columns that are mostly empty"""
    drop = []
    for col in df.columns:
        coverage = df[col].notna().mean()
        if coverage < lim:
            LOG.debug(
                f"Dropping column {col} as it covers {coverage:.1%} "
                f"of the data (limit: {lim:.0%})"
            )
    return drop

def add_rdk_fingerprints(df: pd.DataFrame, thresh : float = 0.05) -> pd.DataFrame:
    """
    Converts the SMILES string to fingerprint
    Args:
        df: pd.DataFrame - dataframe with input features and output that includes SMILES string
        thresh: float - (0-1) threshold for coverage at which to remove a feature.
                        0.05 default (keeps features if they cover 5% of the samples)

    Returns:
        pd.DataFrame - dataframe with SMILES swapped for a fingerprint
    """
    # Convert SMILES to Mol object (+ drop invalid SMILES)
    df.columns = df.columns.str.lower()
    df["mol"] = df["smiles"].apply(_to_mol)
    df = df[df["mol"].notna()].copy()

    # Calculate the molecular descriptors
    desc: list[dict] = df["mol"].apply(_calc_desc).tolist()
    descr_df = pd.DataFrame(desc, index=df.index)

    # Clean infinities / missing
    descr_df = descr_df.replace([np.inf, -np.inf], np.nan)

    # add features to main df
    df = df.join(descr_df, how="inner")

    # remove features/columns that are mostly empty / below threshold
    empty_cols: list = _get_empty_cols(df, thresh)
    empty_cols.extend(["smiles", "mol"])
    df.drop(columns=empty_cols, inplace=True)

    return df

