import argparse
import concurrent.futures as cf
import json
import math
import os
from pathlib import Path

# Keep backend libraries single-threaded as much as possible.
# Set before importing heavy numeric libraries.
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from network import ICNN


EVAL_PAIRS = [
    ("2021bull_to_2022bear", "eval_2021_bull.npy", "eval_2022_bear.npy"),
    ("2024bull_to_2025bear", "eval_2024_bull.npy", "eval_2025_bear.npy"),
]


def parse_k_values(raw: str):
    if raw is None or raw.strip() == "":
        return None
    return {float(x.strip()) for x in raw.split(",") if x.strip()}


def validate_device(device: str) -> str:
    d = device.lower()
    if d == "cuda" and not torch.cuda.is_available():
        raise ValueError("Requested device=cuda but CUDA is not available.")
    if d == "mps":
        mps_ok = hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
        if not mps_ok:
            raise ValueError("Requested device=mps but MPS is not available.")
    if d not in {"cpu", "cuda", "mps"}:
        raise ValueError("device must be one of: cpu, cuda, mps")
    return d


def parse_m1(m1_value):
    if isinstance(m1_value, str) and m1_value.lower() in {"inf", "infty", "infinity"}:
        return float("inf")
    return float(m1_value)


def format_m1(m1_value) -> str:
    m1_float = parse_m1(m1_value)
    if math.isinf(m1_float):
        return "inf"
    return f"{m1_float:g}"


def parse_epoch_from_path(ckpt_path: Path) -> int:
    stem = ckpt_path.stem
    if not stem.startswith("epoch_"):
        return -1
    try:
        return int(stem.split("_", 1)[1])
    except ValueError:
        return -1


def list_checkpoints_for_run(
    model_dir: Path,
    num_epochs: int,
) -> list[Path]:
    cands = sorted(model_dir.glob("epoch_*.pth"))
    if not cands:
        raise FileNotFoundError(f"No checkpoint files found in {model_dir}")

    preferred = model_dir / f"epoch_{num_epochs:04d}.pth"
    if preferred.exists():
        return [preferred]
    return [cands[-1]]


def build_model_from_config(cfg: dict, device: str) -> ICNN:
    return ICNN(
        input_size=int(cfg["input_size"]),
        hidden_size=int(cfg["hidden_size"]),
        output_size=1,
        activation=cfg.get("activation", "softplus_scaled"),
        num_hidden_layers=cfg.get("num_hidden_layers", None),
    ).to(device)


def load_state_into_model(model: ICNN, ckpt_path: Path, device: str) -> None:
    try:
        payload = torch.load(ckpt_path, map_location=device, weights_only=True)
    except TypeError:
        # Backward compatibility for older PyTorch versions without weights_only.
        payload = torch.load(ckpt_path, map_location=device)
    if isinstance(payload, dict) and "model_state_dict" in payload:
        state_dict = payload["model_state_dict"]
    else:
        state_dict = payload
    model.load_state_dict(state_dict)
    model.eval()


def transport_by_gradient(
    model: ICNN,
    x_np: np.ndarray,
    device: str,
    batch_size: int,
) -> np.ndarray:
    outputs = []
    for start in range(0, x_np.shape[0], batch_size):
        x_batch = torch.from_numpy(x_np[start : start + batch_size]).to(device)
        x_batch = x_batch.requires_grad_(True)
        phi = model(x_batch)
        grad = torch.autograd.grad(phi.sum(), x_batch, create_graph=False)[0]
        outputs.append(grad.detach().cpu().numpy())
    return np.concatenate(outputs, axis=0)


def pairwise_sq_dists(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    aa = np.sum(a * a, axis=1, keepdims=True)
    bb = np.sum(b * b, axis=1, keepdims=True).T
    d2 = aa + bb - 2.0 * (a @ b.T)
    return np.maximum(d2, 0.0)


def estimate_rbf_sigma2(z: np.ndarray) -> float:
    d2_zz = pairwise_sq_dists(z, z)
    tri = np.triu_indices(d2_zz.shape[0], k=1)
    upper = d2_zz[tri]
    upper_pos = upper[upper > 0]
    if upper_pos.size == 0:
        return 1.0

    sigma2 = float(np.median(upper_pos))
    if not np.isfinite(sigma2) or sigma2 <= 0:
        sigma2 = float(np.mean(upper_pos))
    if not np.isfinite(sigma2) or sigma2 <= 0:
        sigma2 = 1.0
    return sigma2


def build_rbf_kernels(
    x: np.ndarray,
    y: np.ndarray,
    sigma2: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    d2_xx = pairwise_sq_dists(x, x)
    d2_yy = pairwise_sq_dists(y, y)
    d2_xy = pairwise_sq_dists(x, y)

    Kxx = np.exp(-d2_xx / (2.0 * sigma2))
    Kyy = np.exp(-d2_yy / (2.0 * sigma2))
    Kxy = np.exp(-d2_xy / (2.0 * sigma2))
    return Kxx, Kyy, Kxy


def mmd_unbiased_from_kernels(Kxx: np.ndarray, Kyy: np.ndarray, Kxy: np.ndarray) -> float:
    n = Kxx.shape[0]
    m = Kyy.shape[0]
    if n < 2 or m < 2:
        return float("nan")

    term_xx = (Kxx.sum() - np.trace(Kxx)) / (n * (n - 1))
    term_yy = (Kyy.sum() - np.trace(Kyy)) / (m * (m - 1))
    term_xy = 2.0 * Kxy.mean()
    mmd2 = term_xx + term_yy - term_xy
    return float(max(mmd2, 0.0))


def sample_unit_directions(d: int, n_directions: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    dirs = rng.normal(size=(n_directions, d))
    norms = np.linalg.norm(dirs, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-12)
    dirs = dirs / norms
    return dirs.astype(np.float32)


def sliced_w2_from_projections(proj_x: np.ndarray, proj_y: np.ndarray) -> float:
    n_q = min(proj_x.shape[0], proj_y.shape[0])
    if n_q < 2:
        return float("nan")

    q = (np.arange(n_q, dtype=np.float64) + 0.5) / n_q
    qx = np.quantile(proj_x, q, axis=0)
    qy = np.quantile(proj_y, q, axis=0)
    w2_sq_dir = ((qx - qy) ** 2).mean(axis=0)
    return float(np.sqrt(w2_sq_dir.mean()))


def sliced_w2_distance(
    x: np.ndarray,
    y: np.ndarray,
    directions: np.ndarray,
) -> float:
    proj_x = x @ directions.T
    proj_y = y @ directions.T
    return sliced_w2_from_projections(proj_x, proj_y)


def split_counts(total: int, parts: int) -> list[int]:
    if parts <= 0:
        return []
    q, r = divmod(total, parts)
    return [q + (1 if i < r else 0) for i in range(parts) if (q + (1 if i < r else 0)) > 0]


def bootstrap_mmd_chunk(
    Kxx: np.ndarray,
    Kyy: np.ndarray,
    Kxy: np.ndarray,
    n_bootstrap: int,
    seed: int,
) -> np.ndarray:
    n = Kxx.shape[0]
    m = Kyy.shape[0]
    rng = np.random.default_rng(seed)
    vals = np.empty(n_bootstrap, dtype=np.float64)
    for b in range(n_bootstrap):
        idx_x = rng.integers(0, n, size=n)
        idx_y = rng.integers(0, m, size=m)
        Kxx_b = Kxx[np.ix_(idx_x, idx_x)]
        Kyy_b = Kyy[np.ix_(idx_y, idx_y)]
        Kxy_b = Kxy[np.ix_(idx_x, idx_y)]
        vals[b] = mmd_unbiased_from_kernels(Kxx_b, Kyy_b, Kxy_b)
    return vals


def bootstrap_mmd_sd_from_kernels(
    Kxx: np.ndarray,
    Kyy: np.ndarray,
    Kxy: np.ndarray,
    n_bootstrap: int,
    seed: int,
    n_workers: int = 1,
    progress_cb=None,
) -> float:
    n = Kxx.shape[0]
    m = Kyy.shape[0]
    if n < 2 or m < 2 or n_bootstrap <= 1:
        return float("nan")

    workers = max(1, min(int(n_workers), n_bootstrap))
    counts = split_counts(n_bootstrap, workers)

    if workers == 1:
        vals = bootstrap_mmd_chunk(Kxx, Kyy, Kxy, counts[0], seed)
        if progress_cb is not None:
            progress_cb(counts[0])
        return float(np.std(vals, ddof=1))

    rng = np.random.default_rng(seed)
    seeds = [int(s) for s in rng.integers(0, 2**31 - 1, size=len(counts))]
    vals_parts = []
    with cf.ThreadPoolExecutor(max_workers=workers) as ex:
        fut2count = {
            ex.submit(bootstrap_mmd_chunk, Kxx, Kyy, Kxy, c, s): c
            for c, s in zip(counts, seeds)
        }
        for fut in cf.as_completed(fut2count):
            vals_parts.append(fut.result())
            if progress_cb is not None:
                progress_cb(fut2count[fut])

    vals = np.concatenate(vals_parts, axis=0)
    return float(np.std(vals, ddof=1))


def bootstrap_sw2_chunk(
    proj_x: np.ndarray,
    proj_y: np.ndarray,
    n_bootstrap: int,
    seed: int,
) -> np.ndarray:
    n = proj_x.shape[0]
    m = proj_y.shape[0]
    rng = np.random.default_rng(seed)
    vals = np.empty(n_bootstrap, dtype=np.float64)
    for b in range(n_bootstrap):
        idx_x = rng.integers(0, n, size=n)
        idx_y = rng.integers(0, m, size=m)
        vals[b] = sliced_w2_from_projections(proj_x[idx_x], proj_y[idx_y])
    return vals


def bootstrap_sw2_sd_from_projections(
    proj_x: np.ndarray,
    proj_y: np.ndarray,
    n_bootstrap: int,
    seed: int,
    n_workers: int = 1,
    progress_cb=None,
) -> float:
    n = proj_x.shape[0]
    m = proj_y.shape[0]
    if n < 2 or m < 2 or n_bootstrap <= 1:
        return float("nan")

    workers = max(1, min(int(n_workers), n_bootstrap))
    counts = split_counts(n_bootstrap, workers)

    if workers == 1:
        vals = bootstrap_sw2_chunk(proj_x, proj_y, counts[0], seed)
        if progress_cb is not None:
            progress_cb(counts[0])
        return float(np.std(vals, ddof=1))

    rng = np.random.default_rng(seed)
    seeds = [int(s) for s in rng.integers(0, 2**31 - 1, size=len(counts))]
    vals_parts = []
    with cf.ThreadPoolExecutor(max_workers=workers) as ex:
        fut2count = {
            ex.submit(bootstrap_sw2_chunk, proj_x, proj_y, c, s): c
            for c, s in zip(counts, seeds)
        }
        for fut in cf.as_completed(fut2count):
            vals_parts.append(fut.result())
            if progress_cb is not None:
                progress_cb(fut2count[fut])

    vals = np.concatenate(vals_parts, axis=0)
    return float(np.std(vals, ddof=1))


def write_metric_csv(detail_df: pd.DataFrame, metric_cols: list[str], avg_out: Path):
    grouped = (
        detail_df.groupby(["pair", "k", "M1"], dropna=False)[metric_cols]
        .mean()
        .reset_index()
    )
    counts = (
        detail_df.groupby(["pair", "k", "M1"], dropna=False)
        .size()
        .reset_index(name="n_evals")
    )
    avg_df = grouped.merge(counts, on=["pair", "k", "M1"], how="left")
    avg_df.to_csv(avg_out, index=False)


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate trained finance OT models using MMD and sliced W2."
    )
    parser.add_argument(
        "--models-dir",
        type=str,
        default="model_finance",
        help="Root directory containing trained model configs/checkpoints.",
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default="data_finance/processed/npy",
        help="Directory containing eval_2021_bull.npy / eval_2022_bear.npy / eval_2024_bull.npy / eval_2025_bear.npy.",
    )
    parser.add_argument("--device", type=str, default="cpu", help="Evaluation device.")
    parser.add_argument(
        "--batch-size",
        type=int,
        default=512,
        help="Batch size for transport-gradient computation.",
    )
    parser.add_argument(
        "--n-directions",
        type=int,
        default=1000,
        help="Number of random projection directions for sliced W2.",
    )
    parser.add_argument(
        "--direction-seed",
        type=int,
        default=123,
        help="Random seed for projection directions.",
    )
    parser.add_argument(
        "--output-mmd-csv",
        type=str,
        default="model_finance/evaluation/mmd_average_over_models.csv",
        help="Output CSV path for averaged MMD values.",
    )
    parser.add_argument(
        "--output-sw2-csv",
        type=str,
        default="model_finance/evaluation/sw2_average_over_models.csv",
        help="Output CSV path for averaged sliced W2 values.",
    )
    parser.add_argument(
        "--k-values",
        type=str,
        default="",
        help="Optional comma-separated k values to evaluate (e.g. '-1,1').",
    )
    parser.add_argument(
        "--bootstrap-reps",
        type=int,
        default=1000,
        help="Number of bootstrap resamples for metric standard deviations.",
    )
    parser.add_argument(
        "--bootstrap-seed",
        type=int,
        default=2026,
        help="Random seed used for bootstrap resampling.",
    )
    parser.add_argument(
        "--bootstrap-workers",
        type=int,
        default=4,
        help="Number of parallel workers for bootstrap (default: 4).",
    )
    args = parser.parse_args()

    if args.bootstrap_workers < 1:
        raise ValueError("--bootstrap-workers must be >= 1")

    try:
        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)
    except Exception:
        pass

    device = validate_device(args.device)
    k_filter = parse_k_values(args.k_values)
    models_dir = Path(args.models_dir)
    data_dir = Path(args.data_dir)

    out_paths = [
        Path(args.output_mmd_csv),
        Path(args.output_sw2_csv),
    ]
    for p in out_paths:
        p.parent.mkdir(parents=True, exist_ok=True)

    feature_cols = json.loads((data_dir / "feature_columns.json").read_text())["columns"]
    d = len(feature_cols)
    directions = sample_unit_directions(
        d=d,
        n_directions=args.n_directions,
        seed=args.direction_seed,
    )

    eval_data = {}
    for _, src_name, tgt_name in EVAL_PAIRS:
        eval_data[src_name] = np.load(data_dir / src_name).astype(np.float32)
        eval_data[tgt_name] = np.load(data_dir / tgt_name).astype(np.float32)

    config_paths = sorted(models_dir.glob("k_*/M1_*/model_*/config.json"))
    if not config_paths:
        raise FileNotFoundError(f"No model config files found under {models_dir}")

    jobs = []
    for cfg_path in config_paths:
        cfg = json.loads(cfg_path.read_text())
        k_val = float(cfg["k"])
        if k_filter is not None and k_val not in k_filter:
            continue
        ckpt_paths = list_checkpoints_for_run(
            model_dir=cfg_path.parent,
            num_epochs=int(cfg["num_epochs"]),
        )
        jobs.append((cfg, ckpt_paths))

    if not jobs:
        raise ValueError(
            "No models matched the evaluation filter. "
            "Check --k-values and model_finance contents."
        )

    total_evals = sum(len(ckpts) * len(EVAL_PAIRS) for _, ckpts in jobs)
    rows = []
    pbar = tqdm(total=total_evals, desc="Evaluating pairs", unit="pair")
    seed_rng = np.random.default_rng(args.bootstrap_seed)
    try:
        for cfg, ckpt_paths in jobs:
            model = build_model_from_config(cfg, device=device)
            k_val = float(cfg["k"])
            m1_label = format_m1(cfg["M1"])
            model_num = int(cfg["model_num"])

            for ckpt_path in ckpt_paths:
                load_state_into_model(model=model, ckpt_path=ckpt_path, device=device)
                epoch = parse_epoch_from_path(ckpt_path)

                for pair_name, src_name, tgt_name in EVAL_PAIRS:
                    source = eval_data[src_name]
                    target = eval_data[tgt_name]
                    if source.shape[0] == 0 or target.shape[0] == 0:
                        raise ValueError(
                            f"Empty evaluation array in pair {pair_name}: "
                            f"{src_name}({source.shape[0]}), {tgt_name}({target.shape[0]})"
                        )

                    mapped = transport_by_gradient(
                        model=model,
                        x_np=source,
                        device=device,
                        batch_size=args.batch_size,
                    )
                    sigma2 = estimate_rbf_sigma2(np.vstack([mapped, target]))
                    Kxx, Kyy, Kxy = build_rbf_kernels(mapped, target, sigma2=sigma2)
                    proj_mapped = mapped @ directions.T
                    proj_target = target @ directions.T

                    run_desc = (
                        f"Current run: k={k_val:g}, M1={m1_label}, "
                        f"model={model_num}, {pair_name}"
                    )
                    run_total = 2 * args.bootstrap_reps if args.bootstrap_reps > 1 else 0
                    with tqdm(
                        total=run_total,
                        desc=run_desc,
                        unit="boot",
                        leave=False,
                        position=1,
                        dynamic_ncols=True,
                    ) as run_pbar:
                        mmd_val = mmd_unbiased_from_kernels(Kxx, Kyy, Kxy)
                        mmd_sd = bootstrap_mmd_sd_from_kernels(
                            Kxx=Kxx,
                            Kyy=Kyy,
                            Kxy=Kxy,
                            n_bootstrap=args.bootstrap_reps,
                            seed=int(seed_rng.integers(0, 2**31 - 1)),
                            n_workers=args.bootstrap_workers,
                            progress_cb=run_pbar.update,
                        )

                        sw2_val = sliced_w2_from_projections(proj_mapped, proj_target)
                        sw2_sd = bootstrap_sw2_sd_from_projections(
                            proj_x=proj_mapped,
                            proj_y=proj_target,
                            n_bootstrap=args.bootstrap_reps,
                            seed=int(seed_rng.integers(0, 2**31 - 1)),
                            n_workers=args.bootstrap_workers,
                            progress_cb=run_pbar.update,
                        )

                    rows.append(
                        {
                            "pair": pair_name,
                            "k": k_val,
                            "M1": m1_label,
                            "model_num": model_num,
                            "epoch": epoch,
                            "checkpoint": str(ckpt_path),
                            "mmd_rbf": mmd_val,
                            "mmd_rbf_bootstrap_sd": mmd_sd,
                            "sw2": sw2_val,
                            "sw2_bootstrap_sd": sw2_sd,
                        }
                    )
                    pbar.update(1)
    finally:
        pbar.close()

    if not rows:
        raise ValueError("No evaluation rows were created.")

    detail_df = pd.DataFrame(rows)
    write_metric_csv(
        detail_df=detail_df,
        metric_cols=["mmd_rbf", "mmd_rbf_bootstrap_sd"],
        avg_out=Path(args.output_mmd_csv),
    )
    write_metric_csv(
        detail_df=detail_df,
        metric_cols=["sw2", "sw2_bootstrap_sd"],
        avg_out=Path(args.output_sw2_csv),
    )

    print(f"Saved MMD average CSV to: {args.output_mmd_csv}")
    print(f"Saved SW2 average CSV to: {args.output_sw2_csv}")


if __name__ == "__main__":
    main()
