# Robust Optimal Transport Map Estimation with Sieve Conjugate

This repository provides Python code to reproduce the simulation experiments in **Section 6.1** and **Appendix S1.1** of:

**Robust Optimal Transport Map Estimation with Sieve Conjugate**

## What This Code Reproduces

The simulations compare:
- **Dual-type OT map estimator** (`k = -1.0`, no projection), and
- **Sieve OT map estimator** (`k = 1.0` or `k = 2.0`, projected inner optimization)

across:
- dimensions `d in {10, 20}`,
- sample sizes `n in {100, 300, 500, 1000}`,
- source distributions `P in {normal, t(df=6)}`,
- target map types `{CDF, piecewise_linear, quadratic}`.

## Repository Structure

- `run_simulation.py`: main training script (includes training loop + grid launcher).
- `dataset.py`: synthetic data generator and dataset wrapper.
- `network.py`: ICNN architecture and helper functions.
- `conjugate_optimization.py`: inner-loop convex conjugate optimization routines.
- `analyze_results.ipynb`: evaluation + plotting + LaTeX-table generation notebook.
- `LICENSE`: GNU GPL v3 text.
- `Apache_LICENSE`: Apache 2.0 text for ICNN adaptation notice.

## Requirements

Required third-party libraries:
`torch`, `numpy`, `scipy`, `pandas`, `matplotlib`, and `jupyter` (for running the notebook).

For the finance data pipeline, also install:
`yfinance` and `tqdm`.

Example install command:

```bash
pip install torch numpy scipy pandas matplotlib jupyter yfinance tqdm
```

## How to Run Simulations

From the repository root:

```bash
python run_simulation.py --threads 5
```

Notes:
- `--threads` controls multiprocessing pool size and defaults to `5`.
- Each hyperparameter configuration trains up to 100 models (`model_0.pth` to `model_99.pth`), skipping existing files.
- Current training settings in `run_simulation.py`: ICNN width `16`, batch size `50`, learning rate `1e-3`, outer epochs `500`, and inner epochs `500`.

## Where Trained Files Are Saved

`run_simulation.py` saves model checkpoints with two path patterns:

For the default case `M1 = inf`:

```text
../simulation_results/d=<d>/<measure>_<transform>_n_<n>_k_<k>/model_<idx>.pth
```

For finite `M1`:

```text
../simulation_results/d=<d>/<measure>_<transform>_n_<n>_k_<k>_M1_<M1>/model_<idx>.pth
```

Examples:

```text
../simulation_results/d=10/normal_CDF_n_500_k_1.0/model_17.pth
../simulation_results/d=10/normal_CDF_n_500_k_1.0_M1_4/model_17.pth
```

## How to Analyze Results

Launch:

```bash
jupyter notebook analyze_results.ipynb
```

In the notebook:
1. Set `ROOT` / `orig_root` to your simulation output folder under `../simulation_results/...`.
2. Run evaluation cells to compute `L2_loss` per model and write per-scenario CSV files.
3. Run plotting cells to generate loss boxplots.
4. Run table cells to print LaTeX summary tables.

## License Summary

This repository carries two licensing contexts:
1. Project code under GNU GPL v3 (`LICENSE`).
2. ICNN adaptation notice under Apache 2.0 (`Apache_LICENSE`, and header notes in `network.py`).

## Finance Data Pipeline (Bull/Bear + Scaled Raw Returns)

This repository also includes a finance data pipeline under `finance/` for a real-data OT application.

### Data Source and Assets

- Source: Yahoo Finance daily data via `yfinance`.
- Date range: `2004-01-01` to `2025-12-31`.
- Assets (12 ETFs): `SPY`, `QQQ`, `IWM`, `EFA`, `EEM`, `XLF`, `XLE`, `XLU`, `TLT`, `IEF`, `LQD`, `VNQ`.

### Preprocessing Method

1. Download adjusted close prices.
2. Compute daily log returns.
3. Classify each day as market `bull`/`bear` using `SPY`:
   - `bull` if `close >= MA200` and `60-day return >= 0`;
   - otherwise `bear`.
4. Use scaled raw log returns (`return * 100`) for model inputs.
5. Align to the first date with valid returns for all 12 assets (effective common start is `2004-09-30` because of `VNQ` inception).

### Train/Eval Splits

- Train window: `2004-01-01` to `2020-12-31` (exported as two sets: bull and bear).
- Eval set 1: `2021` bull vs `2022` bear.
- Eval set 2: `2024` bull vs `2025` bear.

### Commands

Run end-to-end:

```bash
python finance/run_pipeline.py
```

Or run each step separately:

```bash
python finance/download_data.py
python finance/preprocess.py
```

Train finance OT models (bull -> bear):

```bash
python run_finance_training.py --threads 2 --device cuda
```

Evaluate trained models with MMD + sliced W2:

```bash
python evaluate_finance.py --device cpu
```

This evaluation script always uses the final checkpoint for each model configuration.
It also computes bootstrap standard deviations on transported/reference vectors
(default `--bootstrap-reps 1000`), with sliced W2 using
`--n-directions 1000` by default.
Bootstrap is parallelized by default with `--bootstrap-workers 4`.

Evaluate only selected `k` values (example: `k=-1` and `k=1`):

```bash
python evaluate_finance.py --device cpu --k-values=-1,1
```

Example with custom bootstrap settings:

```bash
python evaluate_finance.py --device cpu --bootstrap-reps 500 --bootstrap-seed 2026 --bootstrap-workers 4
```

Default training grid:
- Classical OT baseline: `k=-1`, `M1=inf` (no forward truncation).
- Sieve OT variants: `k in {1, 2}`, `M1 in {inf, 2, 4, 8}`.
- Default optimization epochs: outer `500`, inner conjugate `500`.
- Uses independent bull/bear minibatch loaders (no forced pairing); epoch steps are
  `ceil(max(n_bull, n_bear) / batch_size)`.
- Multiprocessing training shows one `tqdm` epoch bar per worker process.
- Device supports `cpu`, `cuda`, and `mps` (if available in your PyTorch build).
- ICNN depth is backward-compatible by default; set `--num-hidden-layers` to use a deeper ICNN.
- Model checkpoints are saved every 10 epochs and at the final epoch.

### Output Files

- Raw data (root-level separate folder):
  - `data_finance/raw/adj_close_prices.csv`
  - `data_finance/raw/log_returns.csv`
- Processed CSV (with `date` and `state` columns):
  - `data_finance/processed/csv/full_scaled_log_returns.csv`
  - `data_finance/processed/csv/train_bull.csv`
  - `data_finance/processed/csv/train_bear.csv`
  - `data_finance/processed/csv/eval_2021_bull.csv`
  - `data_finance/processed/csv/eval_2022_bear.csv`
  - `data_finance/processed/csv/eval_2024_bull.csv`
  - `data_finance/processed/csv/eval_2025_bear.csv`
- Processed NPY (PyTorch-ready float arrays, no dates):
  - `data_finance/processed/npy/train_bull.npy`
  - `data_finance/processed/npy/train_bear.npy`
  - `data_finance/processed/npy/eval_2021_bull.npy`
  - `data_finance/processed/npy/eval_2022_bear.npy`
  - `data_finance/processed/npy/eval_2024_bull.npy`
  - `data_finance/processed/npy/eval_2025_bear.npy`
- Trained models:
  - `model_finance/k_<k>/M1_<M1>/model_<idx>/epoch_<epoch>.pth`
  - Each model folder also includes `config.json` and `train_log.csv`.
- Evaluation output:
  - `model_finance/evaluation/mmd_average_over_models.csv`
    - columns include `mmd_rbf` and `mmd_rbf_bootstrap_sd`
  - `model_finance/evaluation/sw2_average_over_models.csv`
    - columns include `sw2` and `sw2_bootstrap_sd`
