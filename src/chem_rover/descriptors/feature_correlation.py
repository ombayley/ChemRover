#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Author: O. Bayley
Description: Filter RDKit features by correlation with target.
"""

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from src.chem_rover.descriptors.RDKit_Descriptor import load_azo_dataset


def get_top_features(
    x: pd.DataFrame,
    y,
    min_correlation: float = 0.2,
    method: str = "spearman",
    return_filtered_x: bool = False
):
    """
    Compute per-feature correlation with y and keep only features with
    |correlation| >= min_correlation.

    Args:
        x: DataFrame of features, shape (n_samples, n_features)
        y: target array-like, shape (n_samples,)
        min_correlation: threshold for abs(correlation)
        method: "spearman" (default) or "pearson"
        return_filtered_x: if True, also return filtered X

    Returns:
        corr_df: DataFrame with columns [feature_idx, feature_name, correlation]
                 sorted by |correlation| descending
        filtered_x (optional): x reduced to selected features
    """
    y = np.asarray(y).ravel()

    corrs = np.zeros(x.shape[1], dtype=float)

    for i in range(x.shape[1]):
        xi = x.iloc[:, i].to_numpy()

        if method.lower() == "spearman":
            r = spearmanr(xi, y, nan_policy="omit").correlation
        elif method.lower() == "pearson":
            r = np.corrcoef(xi, y)[0, 1]
        else:
            raise ValueError("method must be 'spearman' or 'pearson'")

        if np.isnan(r):
            r = 0.0
        corrs[i] = r

    # Build a results dataframe
    corr_df = pd.DataFrame({
        "feature_idx": np.arange(x.shape[1]),
        "feature_name": x.columns,
        "correlation": corrs
    })

    # Filter by threshold
    corr_df = corr_df[corr_df["correlation"].abs() >= min_correlation]

    # Sort by absolute correlation descending
    corr_df = corr_df.reindex(corr_df["correlation"].abs().sort_values(ascending=False).index)

    if return_filtered_x:
        filtered_x = x.loc[:, corr_df["feature_name"]]
        return corr_df, filtered_x

    return corr_df


if __name__ == "__main__":
    X, y = load_azo_dataset()

    corr_df, X_filtered = get_top_features(
        x=X,
        y=y,
        min_correlation=0.2,
        method="spearman",
        return_filtered_x=True
    )

    print(corr_df.head(20))
    print("Original shape:", X.shape)
    print("Filtered shape:", X_filtered.shape)
