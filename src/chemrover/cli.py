#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Author: O. Bayley
Description: Hydra CLI entry point (`uv run chemrover`) that directs the call to the experiment runner.
"""
import hydra
from pathlib import Path
from omegaconf import DictConfig
from hydra.core.hydra_config import HydraConfig

from chemrover import PROJECT_ROOT
from chemrover.experiment import run


@hydra.main(config_path=str(PROJECT_ROOT / "config"), config_name="config", version_base="1.3")
def main(cfg: DictConfig):
    run(cfg, output_dir=Path(HydraConfig.get().runtime.output_dir))


if __name__ == "__main__":
    main()
