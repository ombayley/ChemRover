#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Author: O. Bayley
Description: **Add Desc**.
"""
import pandas as pd
import numpy as np

from rdkit import Chem
from rdkit.Chem import Descriptors

def load_azo_dataset():
    # --- 1) Load your TSV ---
    df = pd.read_csv(r"C:\Users\OllyBayley\Documents\Repos\personal\MachineLearning\datasets\Azo_lambda_dataset.csv")
    df = df[["SMILES", "lambda"]].dropna()

    # --- 2) SMILES -> Mol, drop invalid SMILES ---
    def to_mol(smiles):
        mol = Chem.MolFromSmiles(smiles)
        return mol

    df["mol"] = df["SMILES"].apply(to_mol)
    df = df[df["mol"].notna()].copy()

    # --- 3) Compute RDKit descriptors ---
    # Descriptors._descList = list of (name, function)
    desc_fns = Descriptors._descList
    desc_names = [name for name, _ in desc_fns]

    def calc_desc(mol):
        vals = []
        for _, fn in desc_fns:
            try:
                vals.append(fn(mol))
            except Exception:
                vals.append(np.nan)
        return vals

    desc_values = df["mol"].apply(calc_desc).to_list()
    X = pd.DataFrame(desc_values, columns=desc_names, index=df.index)

    # Clean infinities / missing
    X = X.replace([np.inf, -np.inf], np.nan)
    X = X.fillna(X.median(numeric_only=True))

    # --- 4) Target ---
    y = df["lambda"].astype(float)
    return X, y

if __name__ == "__main__":
    pass
