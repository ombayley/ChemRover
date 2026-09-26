# ChemRover

> Machine-learning predictors for azobenzene photoswitch properties. Built to be paired with the ChemKlipper repo
> as a part of a 2026 Marsden Grant Proposal. 

ChemRover predicts the absorbance maximum (λ<sub>max</sub>) of azobenzenes from their structure
(SMILES) and measurement context (e.g. solvent). It is built to compare how prediction accuracy
depends on:

- **features**: structure only, or structure plus solvent / source lab
- **encoding**: how a SMILES is turned into numbers (RDKit descriptors, ECFP/FCFP fingerprints)
- **model**: baselines (mean predictor, PLS), XGBoost, random forest, SVM, KNN, Gaussian process and MLP

Once these effects are understood, the same pipeline will be applied to **thermal Z→E relaxation**
(t<sub>½</sub>) prediction.

---

## Setup

Requires Python 3.13 and [uv](https://docs.astral.sh/uv/).

```bash
git clone https://github.com/ombayley/ChemRover.git
cd ChemRover
uv sync            # creates .venv and installs chemrover (editable) plus the dev tools
```

## Running

Every run is configured by [Hydra](https://hydra.cc). With no arguments the CLI runs the defaults
in `config/`:

```bash
uv run chemrover
```

Override any config value from the command line with a dotted path:

```bash
uv run chemrover model=rf                             # model: xgb (default) | rf | svm | knn | gp | pls | mlp | dummy
uv run chemrover data.encoding=ecfp4                  # encoding: rdkit (default) | ecfp4 | ecfp6 | fcfp4 | ..._count
uv run chemrover data.subset=2                        # Griffiths 2022 dataset only
uv run chemrover "data.features=[lambda,SMILES]"      # structure only (no solvent)
uv run chemrover trainer.n_trials=20                  # quicker hyper-parameter search
uv run chemrover model.load_from=outputs/<run>/absorbance_model.json   # reuse a trained model
```

Sweep several values in one command with `-m` (multirun):

```bash
uv run chemrover -m data.subset=1,2,3,4
uv run chemrover -m model=dummy,pls,xgb,rf,svm,knn,gp,mlp
```

Print the fully resolved config without running anything:

```bash
uv run chemrover --cfg job
```

### What a run does

1. **Load** the dataset chosen by `data.subset`, keep the columns in `data.features`, and drop rows with missing values.
2. **Encode** the SMILES with `data.encoding`: `rdkit` (2D descriptors; those defined for fewer than `data.coverage_thresh` of the molecules are dropped) or a Morgan fingerprint such as `ecfp4`, `ecfp6`, `fcfp4` (add `_count` for count vectors, e.g. `ecfp4_count`; length set by `data.fp_bits`, default 2048; bits never set in the dataset are dropped).
3. **Split** the data into train / test / val (`data.test_size`, `data.val_size`, `data.seed`).
4. **Tune** hyper-parameters with Optuna: each trial fits on train and is scored by RMSE on test. The best set is then refit on train, with early stopping on test.
5. **Benchmark** once on the held-out val split, which tuning never sees (RMSE, MAE, R²), and save the model and a parity plot.

### Outputs

```
outputs/
├── 2026-09-26/12-05-19/          # one folder per run
│   ├── .hydra/                   # resolved config + command-line overrides for this run
│   ├── cli.log                   # run log, including the val metrics
│   ├── absorbance_model.json     # trained model (.json for XGB, .joblib for sklearn models)
│   ├── metrics.json              # val RMSE / MAE / R² and split sizes
│   ├── predictions.csv           # val-set y_true, y_pred, residual (+ solvent)
│   └── parity.png                # predicted vs. experimental on the val split
├── multirun/                     # one folder per -m sweep
├── notebook/                     # runs started from notebooks/experiments.ipynb (same layout)
└── optuna/                       # Optuna studies (SQLite)
```

Optuna studies are named after a hash of the `data` and `model` config. Re-running an identical
configuration resumes its study and only runs the trials still missing. Changing the dataset,
features, encoding or model starts a fresh study.

### Comparing settings in a notebook

`notebooks/experiments.ipynb` runs the same `run(cfg)` as the CLI, using the same override strings,
through the helpers in `chemrover.experiment`:

```python
from chemrover.experiment import run_experiment, run_grid, product, parity_grid, load_runs

res = run_experiment("no solvent", ["data.features=[lambda,SMILES]"])     # one run -> RunResult
summary, results = run_grid(product(                                    # every combination
    dataset={"Byadi": "data.subset=1", "Combined": "data.subset=4"},
    solvent={"with": [], "without": "data.features=[lambda,SMILES]"},
))
parity_grid(results)          # parity plots side by side
load_runs()                   # table of every saved run (notebook + CLI)
```

---

## Configuration

```
config/
├── config.yaml        # root: picks one file per group, sets paths and the Hydra output dirs
├── data/abs.yaml      # dataset, feature columns, target, split sizes, encoding
├── model/             # one file per model: class (_target_) and its hyper-parameters
│   ├── xgb.yaml  rf.yaml  svm.yaml  knn.yaml  gp.yaml  pls.yaml  mlp.yaml  dummy.yaml
└── trainer/optuna.yaml  # number of trials, parallelism, study location
```

Each field is commented in its YAML file. All paths are relative to the repo root through the
`${project_root:}` resolver, which is registered in `chemrover/__init__.py`.

## Package layout

```
src/chemrover/
├── cli.py              # Hydra entry point: run(cfg) does load -> tune -> score -> save
├── experiment.py       # notebook helpers: compose configs with overrides, run grids, load past runs
├── datasets/loader.py  # load_df / load_dataset -> (train, test, val) Data splits
├── descriptors/        # SMILES encodings; add_fingerprint() dispatches on data.encoding
├── models/
│   ├── base.py         # ModelBase: the interface every model implements
│   ├── xgb.py          # XGBoost (handles NaNs + categorical solvent natively, early stopping)
│   ├── sklearn_base.py # SklearnModel: shared impute / scale / one-hot pipeline for sklearn models
│   └── rf.py, svm.py, knn.py, gp.py, pls.py, mlp.py, dummy.py
├── trainer/optim_hp.py # model-agnostic Optuna tuning on the test split
├── plotting/           # parity plots, parity grids (and chemical-space embeddings)
└── util/setup_logging.py
```

## Extending

### Adding an encoding
1. Write a function in `src/chemrover/descriptors/` that takes a DataFrame with a lower-case `smiles` column and returns it with `smiles` replaced by numeric feature columns, keeping the other columns (see `ecfp.py`).
2. Add a branch for its name in `descriptors/loader.py::add_fingerprint`.
3. Select it with `uv run chemrover data.encoding=<name>`.

### Adding a model

**Any scikit-learn regressor** (e.g. Ridge) takes about 15 lines. Copy `models/knn.py`:

```python
from sklearn.linear_model import Ridge
from .sklearn_base import SklearnModel

class RidgeModel(SklearnModel):
    ESTIMATOR = Ridge

    @property
    def PARAM_LIMS(self):   # (low, high[, 'log']) ranges or [choices]
        return {'alpha': (1e-3, 1e3, 'log')}
```

Then add `config/model/ridge.yaml` with `_target_: chemrover.models.ridge.RidgeModel`, a `name`,
`load_from: null`, `scale_features` / `scale_target` and any fixed regressor arguments. Select it with
`uv run chemrover model=ridge`. `SklearnModel` supplies the preprocessing (median imputation,
optional standard scaling, one-hot solvent), `.joblib` saving/loading and `clone`/`set_params`.

Models with nothing to tune return an empty `PARAM_LIMS`, and the trainer then skips Optuna and fits once.
`Dummy` has no parameters, and `GP` fits its kernel hyper-parameters by marginal likelihood (choose the
kernel with `model.kernel=rbf|matern32|matern52|rq`). A model whose config values need translating
before reaching sklearn (e.g. the GP's kernel name, the MLP's `n_layers` x `n_units`) overrides `_estimator()`.

**Any other library:** subclass `ModelBase` directly, as `models/xgb.py` does. Implement
`X_test, y_test`, `predict`, `score`, `clone`, `save`, `load` and `set_params`, plus
the `name`, `PARAM_LIMS` and `FIXED_PARAMS` properties.

---

## Data

Raw CSVs live in `data/raw/`. The `data.subset` setting chooses between them:

| `subset` | File | Source |
|---|---|---|
| 1 | `1_J_Chem_Inform_17_42_2025_Byadi.csv` | Byadi *et al.*, J. Cheminform. **17**, 42 (2025) |
| 2 | `2_Chem_Sci_13_45_2022_Griffiths.csv` | Griffiths *et al.*, Chem. Sci. **13** (2022) |
| 3 | `3_Phys_Chem_Chem_Phys_2007_9_18.csv` | Phys. Chem. Chem. Phys. **9** (2007) |
| 4 | `4_Combined.csv` | Sets 1–3 concatenated (SMILES, λ<sub>max</sub>, solvent) |
| – | `5_azobenzene_thermal_relaxation_merged.csv` | Thermal relaxation (t<sub>½</sub>) data; not yet wired into the loader |

> Note: set 4 contains repeated molecules (1,363 rows, 1,239 unique structures), so a random split
> can place the same molecule in both train and val.

---

## Results

>Dataset split into 700 train / 93 test (hyper-parameter tuning) / 141 val (final benchmark)

Currently, the model predictions are very good with most models providing r2 >0.9. XGB appears to edge out the competition 
when assessed over multiple runs with varying datasets, however XGB, SVM, RF, GPs and KNNs all provide predictions that 
are roughly within the margin of error of each-other, making it difficult to establish a single clear winner. 
MLP and PLS both underperform compared to the other models which is expected for a small dataset with non-linear responses.

![all_models_direct_comparison.png](reports/assets/all_models_direct_comparison.png)

![top_models_learning_curve_overlay.png](reports/assets/top_models_learning_curve_overlay.png)

see the [Absorbance Results](reports/abs_results.md) file for more details and rough summary of the prediction of the azobenzene 
absorbance bands.

---

## Road Map

### 1. Encoding Sweep
Having seen the absorbances can be rather accurately predicted from the RDkit FP encoded structures, I want to sweep
a variety of different molecular encoders to test the effect.

[scikit-fingerprint](https://github.com/MLCIL/scikit-fingerprints) provides a large range of chemical encoders so will 
do the sweep using this library. Need to solve numpy, pandas and rdkit dependency clash as the scikit-fp pkg uses 
some old versions as dependencies.

### 2. Data Examination

Based on the parity plots there appear to be some data points are consistently the more difficult to predict 
for multiple models. Want to add outlier detection to flag consistent outlier datapoints for further investigation

### 3. Feature Space Examination

Want to implement feature space examination (K-Means Clustering, UMAP and PCA) to identify the current 
feature space coverage.

### 4. Thermal Relaxation

Once the accuracy of the absorbance models is at its max from the current assessments will shift to use the same 
pipeline to examine the thermal relaxation rates.

---

## License

MIT; see [LICENSE](LICENSE).
