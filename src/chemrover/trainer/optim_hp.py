#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Author: O. Bayley
Description: **Add Desc**.
"""
import optuna
from sklearn.model_selection import KFold
from chemrover.models import ModelBase

def get_params(trial, model):
    params = {}
    for key, val in model.PARAM_LIMS.items():
        low, high, *opts = val
        log = 'log' in opts
        if isinstance(low, int) and isinstance(high, int):
            params[key] = trial.suggest_int(key, low, high, log=log)
        params[key] = trial.suggest_float(key, low, high, log=log)
    return params | model.FIXED_PARAMS

def optimise_hyper_params(cfg, model, X_train, y_train, X_test, y_test):
    def objective(trial):
        params = get_params(trial, model)
        model.set_params(**params)
        model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)
        trial.set_user_attr("best_iteration", model.best_iteration)
        scores = model.score(y_test, model.predict(X_test))
        return scores["rmse"]

    db_path = cfg.save_path / f"{model.name}_study.db"
    db_path.parent.mkdir(parents=True, exist_ok=True)
    study = optuna.create_study(
        study_name=f"{model.name}_optim",
        storage=f"sqlite:///{db_path.as_posix()}",
        direction='minimize',
        load_if_exists=True,
    )
    study.optimize(objective, n_trials=cfg.n_trials, show_progress_bar=cfg.show_progress_bar, n_jobs=cfg.njobs)
    model.set_params(study.best_params)
    model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)
    return model
#
# def hyperparameter_tune(model: ModelBase):
#     def objective(trial):
#         params = get_params(trial, model)
#
#         # Also calculates r2 like 'cross_val_score' but this includes a pruning hook allowing early termination
#         kf = KFold(n_splits=5, shuffle=True, random_state=0)
#         scores, rounds = [], []
#         for i, (train, val) in enumerate(kf.split(X_train)):
#             model.set_params(**params)
#             model.fit(X_train.iloc[train], y_train.iloc[train],
#                       eval_set=[(X_train.iloc[val], y_train.iloc[val])], verbose=False)
#
#
#             scores.append(r2_score(y_train.iloc[val], model.predict(X_train.iloc[val])))
#             rounds.append(model.best_iteration)
#
#             trial.report(np.mean(scores), i)
#             if trial.should_prune():
#                 raise optuna.TrialPruned()
#
#         trial.set_user_attr("best_n_estimators", int(np.mean(rounds)))
#         return float(np.mean(scores))
#
#
#     db_path = SAVE_PATH / f"{name}_hp_opt_study.db"
#     db_path.parent.mkdir(parents=True, exist_ok=True)
#
#     study = optuna.create_study(
#         study_name=f"{name}_hp_opt",
#         storage=f"sqlite:///{db_path.as_posix()}",
#         direction='maximize',
#         load_if_exists=True,
#         pruner=optuna.pruners.MedianPruner(n_startup_trials=10, n_warmup_steps=1),
#     )
#     study.optimize(objective, n_trials=100, show_progress_bar=True, n_jobs=-1)
#
#     best_params = study.best_params
#     final_model = XGBRegressor(**best_params, **CAT_KWARGS)
#     final_model.fit(X_train, y_train)  # Must fit before saving
#     file = SAVE_PATH / f"{name}_model.json"
#     final_model.save_model(file)
#     LOG.info(f"\nBest parameters: {best_params}")
#     return final_model
