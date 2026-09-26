#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Author: O. Bayley
Description: Loads an azobenzene dataset from the raw CSVs, encodes the SMILES and splits it into
             train / val / test sets according to the `data` config group.
"""
import pandas as pd
from pathlib import Path
from dataclasses import dataclass
from sklearn.model_selection import train_test_split
from chemrover.descriptors import add_fingerprint

# Known datasets, keyed by the `data.subset` config value
NAME_MAP = {
    1: "1_J_Chem_Inform_17_42_2025_Byadi",
    2: "2_Chem_Sci_13_45_2022_Griffiths",
    3: "3_Phys_Chem_Chem_Phys_2007_9_18",
    4: "4_Combined",
    5: "5_azobenzene_thermal_relaxation_merged"
}


@dataclass
class Data:
    """Feature matrix and target for one split."""
    X: pd.DataFrame
    y: pd.Series


def load_df(cfg) -> pd.DataFrame:
    """
    Loads one of the raw azobenzene datasets and trims it to the configured rows and columns.
    Column names are lower-cased (in the data and in the config lists) so the config is case-insensitive.
    Rows missing a value in any `required` or feature column are dropped, then at most `max_samples`
    rows are kept (seeded), so variants that only differ in `features` use identical rows.
    Args:
        cfg: DictConfig - the `data` config group (uses path, subset, features, required, max_samples,
             target, categorical_features, seed)
    Returns:
        pd.DataFrame - the target plus the requested feature columns
    """
    if cfg.subset not in NAME_MAP:
        raise ValueError(f"subset {cfg.subset} not available (known datasets: {list(NAME_MAP)})")
    df = pd.read_csv(Path(cfg.path) / f"{NAME_MAP[cfg.subset]}.csv")
    df.columns = df.columns.str.lower()

    # keep the target plus the requested features (target first, no duplicates)
    target = cfg.target.lower()
    columns = list(dict.fromkeys([target, *(f.lower() for f in cfg.features)]))
    required = [c.lower() for c in cfg.get("required") or []]
    missing = [c for c in {*columns, *required} if c not in df.columns]
    if missing:
        raise KeyError(f"columns {missing} not found in dataset {cfg.subset} (available: {list(df.columns)})")
    df = df.dropna(subset=list({*columns, *required}))[columns].copy()

    # optional cap on the number of rows (same seed -> same rows for every feature variant)
    if cfg.get("max_samples") and len(df) > cfg.max_samples:
        df = df.sample(n=cfg.max_samples, random_state=cfg.seed)

    # convert categorical feature strings to category dtype (type needed for XGBoost to use categoricals)
    for feature in cfg.categorical_features:
        if feature.lower() in df.columns:
            df[feature.lower()] = df[feature.lower()].astype("category")

    # ensure regression target is float
    df[target] = df[target].astype(float)
    return df


def load_dataset(cfg) -> tuple[Data, Data, Data]:
    """
    Loads, encodes and splits a dataset.
    Split roles: train is used to fit, test for hyper-parameter tuning / early stopping,
    and val is held out for the final benchmark only.
    Args:
        cfg: DictConfig - the `data` config group
    Returns:
        tuple[Data, Data, Data] - (train, test, val)
    """
    df = load_df(cfg)
    df = add_fingerprint(df, method=cfg.encoding, thresh=cfg.coverage_thresh, n_bits=cfg.get("fp_bits", 2048))

    target = cfg.target.lower()
    X = df.drop(columns=[target])
    y = df[target]

    # hold out test + val together, then divide that held-out part into test (tuning) and val (benchmark)
    X_train, X_vt, y_train, y_vt = train_test_split(X, y, test_size=(cfg.test_size+cfg.val_size), random_state=cfg.seed)
    # val_size is a fraction of the full dataset, so rescale it to val's share of the held-out rows
    val_frac = cfg.val_size / (cfg.test_size+cfg.val_size)
    X_test, X_val, y_test, y_val = train_test_split(X_vt, y_vt, test_size=val_frac, random_state=cfg.seed)

    # learning curves: keep only part of the training split; test and val stay fixed. Taking the first
    # n rows of one seeded shuffle nests the subsets (the 25% sample is inside the 50% sample).
    if cfg.get("train_fraction", 1.0) < 1.0:
        keep = X_train.sample(frac=1.0, random_state=cfg.seed).index[:max(1, round(len(X_train) * cfg.train_fraction))]
        X_train, y_train = X_train.loc[keep], y_train.loc[keep]

    return Data(X_train, y_train), Data(X_test, y_test), Data(X_val, y_val)
