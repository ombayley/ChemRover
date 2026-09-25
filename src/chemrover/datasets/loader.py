#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Author: O. Bayley
Description: Simple dataset loader.
"""
import pandas as pd
from pathlib import Path
from dataclasses import dataclass
from sklearn.model_selection import train_test_split
from chemrover.descriptors import add_fingerprint

@dataclass
class Data:
    X: pd.DataFrame
    y: pd.Series

def load_df(cfg) -> pd.DataFrame:
    """
    Loads 1 of the available azobenzene datasets (id 1-4) and returns the pd.Dataframe with the
    specified keys (Default: SMILES and lambda). Drops any rows missing data for the specified keys.
    Args:
        set_id: int - dataset id from 1-3
        keys: list - list of keys to include in the dataframe. Defaults to ["SMILES", "lambda"]
    Returns:
        pd.DataFrame - dataframe with specified keys (Default: SMILES and lambda)
    """
    # Known datasets keyed for convenience
    name_map = {
        1: "1_J_Chem_Inform_17_42_2025_Byadi",
        2: "2_Chem_Sci_13_45_2022_Griffiths",
        3: "3_Phys_Chem_Chem_Phys_2007_9_18",
        4: "4_Combined"
    }

    # open dataset as df
    if cfg.subset not in name_map.keys():
        raise ValueError(f"set_id {cfg.subset} not available (known datasets: {name_map.keys()})")
    p = Path(cfg.path) / f"{name_map[cfg.subset]}.csv"
    df = pd.read_csv(filepath_or_buffer=p)

    # keep only desired feature columns
    df = df[cfg.features]

    # convert categorical features strings to category dtype (type needed for XGBoost to use categoricals)
    for feature in cfg.categorical_features:
        if feature in df.columns:
            df[feature] = df[feature].astype("category")

    # ensure regression target is float
    if cfg.target in df.columns:
        df[cfg.target] = df[cfg.target].astype(float)

    # return trimmed dataset
    return df[cfg.features].dropna()


def load_dataset(cfg):
    df = load_df(cfg)
    df = add_fingerprint(df, method=cfg.encoding)

    # Split X and y from df
    X = df.drop(columns=[cfg.target])
    y = df[cfg.target]

    # Get train test val splits
    train_X, test_X, train_y, test_y = train_test_split(X, y, test_size=cfg.test_size, random_state=cfg.seed)
    train_X, val_X, train_y, val_y = train_test_split(X, y, test_size=cfg.val_size, random_state=cfg.seed)
    train = Data(train_X, train_y)
    test = Data(test_X, test_y)
    val = Data(val_X, val_y)

    return train, test, val