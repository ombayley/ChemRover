#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Author: O. Bayley
Description: Runs the ChemRover pipeline (load -> tune -> score -> save) and compares runs.
             `run(cfg)` is what the CLI calls. `run_experiment` / `run_grid` let a notebook change the
             defaults with the same override strings as the command line, e.g.
                 run_experiment("no solvent", ["data.features=[lambda,SMILES]"])
"""
import hashlib
import itertools
import json
import logging
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional, Sequence

import hydra
import matplotlib.pyplot as plt
import pandas as pd
from hydra import compose, initialize_config_dir
from omegaconf import DictConfig, OmegaConf

from chemrover import CONFIG_DIR, OUTPUT_DIR
from chemrover.datasets import load_dataset
from chemrover.models import ModelBase
from chemrover.plotting import plot_parity, plot_parity_grid
from chemrover.trainer import optimise_hyper_params
from chemrover.util import setup_logging

LOG = logging.getLogger(__name__)


@dataclass
class RunResult:
    """Everything produced by one run."""
    cfg: DictConfig
    model: ModelBase
    metrics: dict                # test rmse / mae / r2 + split sizes
    predictions: pd.DataFrame    # val split: y_true, y_pred, residual (+ categorical features)
    output_dir: Optional[Path] = None


def _study_name(cfg: DictConfig) -> str:
    """One Optuna study per data + model config: identical re-runs resume, anything else starts fresh."""
    data = {k: v for k, v in OmegaConf.to_container(cfg.data, resolve=True).items() if k != "path"}
    model = {k: v for k, v in OmegaConf.to_container(cfg.model, resolve=True).items() if k != "load_from"}
    digest = hashlib.sha1(json.dumps([data, model], sort_keys=True).encode()).hexdigest()[:8]
    return f"{cfg.model.name}_{digest}"


def run(cfg: DictConfig, output_dir: Optional[Path] = None) -> RunResult:
    """
    Runs one experiment: fit on train, tune on test, benchmark on the val split.
    Args:
        cfg: DictConfig - the composed root config
        output_dir: Path | None - where to save run.log, the model, metrics.json, predictions.csv and
                    parity.png (nothing saved if None)
    Returns:
        RunResult - config, fitted model, test metrics and test predictions
    """
    setup_logging(out_path=output_dir)
    train, test, val = load_dataset(cfg.data)

    model = hydra.utils.instantiate(cfg.model)
    if not model.is_fitted:   # a model given via model.load_from skips tuning
        model = optimise_hyper_params(cfg.trainer, model, train.X, train.y, test.X, test.y,
                                      study_name=_study_name(cfg))

    y_pred = model.predict(val.X)
    metrics = model.score(val.y, y_pred) | {"n_train": len(train.y), "n_val": len(val.y),
                                             "n_test": len(test.y), "n_features": train.X.shape[1]}
    LOG.info(f"Val metrics: {metrics}")
    predictions = (pd.DataFrame({"y_true": val.y, "y_pred": y_pred}, index=val.y.index)
                   .assign(residual=lambda d: d.y_pred - d.y_true)
                   .join(val.X.select_dtypes("category")))

    if output_dir is not None:
        output_dir = Path(output_dir)
        model.save(output_dir)
        predictions.to_csv(output_dir / "predictions.csv")
        (output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
        fig, *_ = plot_parity(val.y, y_pred, title=f"{model.name} - dataset {cfg.data.subset} (val set)",
                              save_path=str(output_dir / "parity.png"))
        plt.close(fig)
    return RunResult(cfg, model, metrics, predictions, output_dir)


def compose_cfg(overrides: Sequence[str] = ()) -> DictConfig:
    """The root config as the CLI builds it, with CLI-style overrides applied (a repeated key: last wins)."""
    with initialize_config_dir(config_dir=str(CONFIG_DIR), version_base="1.3"):
        return compose(config_name="config", overrides=list(overrides))


def run_experiment(label: str, overrides: Sequence[str] = (), save: bool = True) -> RunResult:
    """
    Runs the pipeline with CLI-style overrides on top of the defaults.
    Args:
        label: str - short name, used for the output folder outputs/notebook/<date>/<time>_<label>/
        overrides: list[str] - e.g. ["model=rf", "data.subset=2"]
        save: bool - save the run (plus .hydra/config.yaml + overrides.yaml, as a CLI run does)
    """
    cfg = compose_cfg(overrides)
    output_dir = None
    if save:
        output_dir = OUTPUT_DIR / "notebook" / f"{datetime.now():%Y-%m-%d/%H-%M-%S}_{re.sub(r'\W+', '_', label)}"
        (output_dir / ".hydra").mkdir(parents=True)
        OmegaConf.save(cfg, output_dir / ".hydra" / "config.yaml", resolve=True)
        OmegaConf.save(list(overrides), output_dir / ".hydra" / "overrides.yaml")
    LOG.info(f"=== {label}: {list(overrides) or 'defaults'}")
    return run(cfg, output_dir)


def product(**axes: dict[str, str | list[str]]) -> dict:
    """
    Every combination of named axes, for run_grid. Each axis maps a label to override(s):
        product(model={"XGB": "model=xgb", "RF": "model=rf"},
                solvent={"with": [], "without": "data.features=[lambda,SMILES]"})
        -> {("XGB", "with"): ["model=xgb"], ("XGB", "without"): ["model=xgb", "data.features=..."], ...}
    With one axis the keys are plain labels.
    """
    variants = {}
    for combo in itertools.product(*(ax.items() for ax in axes.values())):
        key = tuple(label for label, _ in combo)
        variants[key[0] if len(key) == 1 else key] = [o for _, ovr in combo
                                                      for o in ([ovr] if isinstance(ovr, str) else ovr)]
    return variants


def _label(key) -> str:
    """Display label for a run_grid key (tuple keys from product are joined)."""
    return " | ".join(key) if isinstance(key, tuple) else str(key)


def run_grid(variants: dict, common: Sequence[str] = ()) -> tuple[pd.DataFrame, dict]:
    """
    Runs every {label: overrides} variant, with ``common`` overrides applied first.
    Returns:
        (metrics table indexed by label - one index level per product() axis, {label: RunResult})
    """
    results = {key: run_experiment(_label(key), [*common, *ovr]) for key, ovr in variants.items()}
    return pd.DataFrame.from_dict({k: r.metrics for k, r in results.items()}, orient="index"), results


def parity_grid(results: dict, **kwargs):
    """Parity plots of run_grid results side by side (kwargs go to plotting.plot_parity_grid)."""
    return plot_parity_grid({_label(k): (r.predictions.y_true, r.predictions.y_pred)
                             for k, r in results.items()}, **kwargs)


def load_runs(root: Path = OUTPUT_DIR) -> pd.DataFrame:
    """Every saved run (CLI and notebook) with its overrides and test metrics, newest first."""
    rows = []
    for f in Path(root).rglob("metrics.json"):
        overrides = f.parent / ".hydra" / "overrides.yaml"
        rows.append({"time": datetime.fromtimestamp(f.stat().st_mtime),
                     "run": f.parent.relative_to(root).as_posix(),
                     "overrides": " ".join(OmegaConf.load(overrides)) if overrides.exists() else "",
                     **json.loads(f.read_text())})
    return pd.DataFrame(rows).sort_values("time", ascending=False, ignore_index=True) if rows else pd.DataFrame()
