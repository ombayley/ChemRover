#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Author: O. Bayley
Description: **Add Desc**.
"""
import hydra
import matplotlib.pyplot as plt
from omegaconf import DictConfig
from.plotting import plot_parity
from .datasets import load_dataset
from .trainer import optimise_hyper_params

@hydra.main(config_path="../../config", config_name="config", version_base="1.3")
def main(cfg : DictConfig):
    model = hydra.utils.instantiate(cfg.model)
    train, test, val= load_dataset(cfg.data)
    model = optimise_hyper_params(cfg, model, train.X, train.y, test.X, test.y)
    model.fit(train.X, train.y, eval_set=[(test.X, test.y)], verbose=False)
    model.save()

    fig = plt.figure(figsize=(13, 15), facecolor="white")
    subfigs = fig.subfigures(2, 2, wspace=0.03, hspace=0.03)
    for sfig, set_id in zip(subfigs.ravel(), (1, 2, 3, 4)):
        plot_parity(
            val.y, model[val.X],
            title=f"Dataset #{set_id}",
            target_fig=sfig,
        )

    scores = model.score(val.y, model[val.X])
    print(scores)

if __name__ == "__main__":
    main()
