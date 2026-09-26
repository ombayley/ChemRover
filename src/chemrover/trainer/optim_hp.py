#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Author: O. Bayley
Description: Optuna hyper-parameter optimisation for any ModelBase model.
             Trials are scored by RMSE on the test split; the val split is never seen here.
"""
import logging
import optuna
from pathlib import Path
from typing import Optional
from chemrover.models import ModelBase

LOG = logging.getLogger(__name__)


def get_params(trial: optuna.Trial, model: ModelBase) -> dict:
    """
    Samples one set of hyper-parameters from the model's PARAM_LIMS search space.
    Each entry is either a tuple range (low, high[, 'log']) - int bounds are sampled as ints,
    'log' samples on a log scale - or a list of choices, e.g. ['uniform', 'distance'].
    Args:
        trial: optuna.Trial - the current trial
        model: ModelBase - model providing PARAM_LIMS and FIXED_PARAMS
    Returns:
        dict - sampled params merged with the model's FIXED_PARAMS
    """
    params = {}
    for key, val in model.PARAM_LIMS.items():
        if isinstance(val, list):
            params[key] = trial.suggest_categorical(key, val)
            continue
        low, high, *opts = val
        log = 'log' in opts
        if isinstance(low, int) and isinstance(high, int):
            params[key] = trial.suggest_int(key, low, high, log=log)
        else:
            params[key] = trial.suggest_float(key, low, high, log=log)
    return params | model.FIXED_PARAMS


def optimise_hyper_params(
        cfg,
        model: ModelBase,
        X_train, y_train,
        X_test, y_test,
        study_name: Optional[str] = None,
) -> ModelBase:
    """
    Tunes the model's hyper-parameters with Optuna and returns a model fitted with the best set.
    The study is stored as SQLite in ``cfg.study_dir`` and resumed if it already exists, so
    ``study_name`` should identify the data + model settings (see experiment._study_name).
    Models with an empty PARAM_LIMS (e.g. Dummy, GP) skip Optuna and are fitted once as configured.
    Args:
        cfg: DictConfig - the `trainer` config group
        model: ModelBase - template model; each trial trains a fresh clone of it
        X_train, y_train: training split
        X_test, y_test: test split used to score trials (and for early stopping)
        study_name: str | None - Optuna study name (defaults to '<model.name>_optim')
    Returns:
        ModelBase - a new model fitted on the training split with the best parameters found
    """
    if not model.PARAM_LIMS:
        LOG.info(f"'{model.name}' has no search space; fitting once with the configured params")
        fitted = model.clone()
        fitted.fit(X_train, y_train, X_test, y_test)
        return fitted

    def objective(trial):
        # a fresh clone per trial: trials run in parallel threads and must not share a model
        candidate = model.clone(**get_params(trial, model))
        candidate.fit(X_train, y_train, X_test, y_test)
        if candidate.best_iteration is not None:
            trial.set_user_attr("best_iteration", candidate.best_iteration)
        return candidate.score(y_test, candidate.predict(X_test))["rmse"]

    study_name = study_name or f"{model.name}_optim"
    db_path = Path(cfg.study_dir) / f"{study_name}.db"
    db_path.parent.mkdir(parents=True, exist_ok=True)
    study = optuna.create_study(
        study_name=study_name,
        storage=f"sqlite:///{db_path.as_posix()}",
        direction='minimize',
        load_if_exists=True,
    )

    # only run the trials still missing from a resumed study
    n_done = len(study.get_trials(states=(optuna.trial.TrialState.COMPLETE,)))
    n_remaining = max(0, cfg.n_trials - n_done)
    LOG.info(f"Study '{study_name}': {n_done} trials done, running {n_remaining} more")
    if n_remaining:
        study.optimize(objective, n_trials=n_remaining, show_progress_bar=cfg.show_progress_bar, n_jobs=cfg.n_jobs)

    LOG.info(f"Best test RMSE {study.best_value:.3f} with params {study.best_params}")
    best = model.clone(**study.best_params)
    best.fit(X_train, y_train, X_test, y_test)
    return best
