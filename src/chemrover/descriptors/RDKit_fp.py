#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Author: O. Bayley
Description: RDKit molecular descriptor encoding. Each SMILES is converted to the full set of
             ~200 RDKit 2D descriptors (Descriptors.CalcMolDescriptors: MolWt, TPSA, logP, ring
             counts, charge/VSA descriptors, ...). Despite the module name these are descriptors,
             not bit-vector fingerprints.
"""
import logging
import pandas as pd
import numpy as np
from typing import Optional
from rdkit import Chem
from rdkit.Chem import Descriptors
LOG = logging.getLogger(__name__)

def _to_mol(smiles: str) -> Optional[Chem.Mol]:
    """Convert a SMILES string to an RDKit Mol (None if the SMILES is invalid)."""
    return Chem.MolFromSmiles(smiles)


def _calc_desc(mol: Chem.Mol) -> dict:
    """Calculate every RDKit descriptor for ``mol`` as {descriptor_name: value}."""
    return Descriptors.CalcMolDescriptors(mol)


def _get_empty_cols(df: pd.DataFrame, lim: float) -> list[str]:
    """Return the columns whose non-NaN coverage is below ``lim`` (0-1)."""
    drop = []
    for col in df.columns:
        coverage = df[col].notna().mean()
        if coverage < lim:
            LOG.debug(
                f"Dropping column {col} as it covers {coverage:.1%} "
                f"of the data (limit: {lim:.0%})"
            )
            drop.append(col)
    return drop


def add_rdk_fingerprints(df: pd.DataFrame, thresh: float = 0.9) -> pd.DataFrame:
    """
    Replaces the 'smiles' column with the full set of RDKit molecular descriptors.
    Rows with invalid SMILES are dropped. The input dataframe is not modified.
    Args:
        df: pd.DataFrame - dataframe with a lower-case 'smiles' column plus any other features/target
        thresh: float - (0-1) minimum non-NaN coverage for a descriptor to be kept.
                        0.9 default (keeps descriptors that are defined for >= 90% of molecules)

    Returns:
        pd.DataFrame - the other input columns joined with the descriptor columns
    """
    # Convert SMILES to Mol object (+ drop invalid SMILES)
    mols = df["smiles"].apply(_to_mol)
    df = df[mols.notna()].drop(columns=["smiles"])
    mols = mols[mols.notna()]

    # Calculate the molecular descriptors and clean infinities
    descr_df = pd.DataFrame(mols.apply(_calc_desc).tolist(), index=mols.index)
    descr_df = descr_df.replace([np.inf, -np.inf], np.nan)

    # remove descriptors that are mostly empty / below threshold
    descr_df = descr_df.drop(columns=_get_empty_cols(descr_df, thresh))

    return df.join(descr_df, how="inner")
