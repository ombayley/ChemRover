#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Author: O. Bayley
Description: Morgan / extended-connectivity fingerprint encoding (ECFP and FCFP).
             Each SMILES becomes a fixed-length vector recording which circular substructures (every
             atom plus its neighbourhood out to `radius` bonds) are present, hashed into `n_bits` slots.
             ECFP4 = radius 2, ECFP6 = radius 3 (the number is the neighbourhood diameter).
             FCFP describes atoms by pharmacophoric role (donor, acceptor, aromatic, halogen, ...)
             instead of exact element/charge, so chemically similar groups share bits.
             Bit vectors record presence; count vectors record how often each substructure occurs.
"""
import logging
import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator

LOG = logging.getLogger(__name__)


def add_morgan_fingerprints(
        df: pd.DataFrame,
        radius: int = 2,
        n_bits: int = 2048,
        counts: bool = False,
        features: bool = False,
) -> pd.DataFrame:
    """
    Replaces the 'smiles' column with a Morgan fingerprint. Rows with invalid SMILES are dropped,
    as are bits that are never set anywhere in the dataset. The input dataframe is not modified.
    Args:
        df: pd.DataFrame - dataframe with a lower-case 'smiles' column plus any other features/target
        radius: int - neighbourhood radius in bonds (2 -> ECFP4, 3 -> ECFP6)
        n_bits: int - fingerprint length
        counts: bool - count vector (occurrences) instead of a bit vector (presence)
        features: bool - FCFP (pharmacophoric atom features) instead of ECFP (exact atom types)
    Returns:
        pd.DataFrame - the other input columns joined with the fingerprint columns, e.g. 'ecfp4_17'
    """
    mols = df["smiles"].apply(Chem.MolFromSmiles)
    df = df[mols.notna()].drop(columns=["smiles"])
    mols = mols[mols.notna()]

    invariants = {"atomInvariantsGenerator": rdFingerprintGenerator.GetMorganFeatureAtomInvGen()} if features else {}
    generator = rdFingerprintGenerator.GetMorganGenerator(radius=radius, fpSize=n_bits, **invariants)
    to_array = generator.GetCountFingerprintAsNumPy if counts else generator.GetFingerprintAsNumPy

    prefix = f"{'fcfp' if features else 'ecfp'}{2 * radius}"
    fp = pd.DataFrame(np.stack([to_array(m) for m in mols]), index=mols.index,
                      columns=[f"{prefix}_{i}" for i in range(n_bits)])
    fp = fp.loc[:, fp.any()]   # bits never set in this dataset carry no information
    LOG.debug(f"{prefix}{'_count' if counts else ''}: {fp.shape[1]} of {n_bits} bits used")
    return df.join(fp)
