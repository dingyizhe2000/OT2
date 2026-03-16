import argparse
import json
import math
from pathlib import Path

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


def mmd_rbf_unbiased(x: np.ndarray, y: np.ndarray) -> float:
    if x.shape[0] < 2 or y.shape[0] < 2:
        return float("nan")

    z = np.vstack([x, y])
    d2_zz = pairwise_sq_dists(z, z)
    tri = np.triu_indices(d2_zz.shape[0], k=1)
    upper = d2_zz[tri]
    upper_pos = upper[upper > 0]
    if upper_pos.size == 0:
        sigma2 = 1.0
    else:
        sigma2 = float(np.median(upper_pos))
        if not np.isfinite(sigma2) or sigma2 <= 0:
            sigma2 = float(np.mean(upper_pos))
        if not np.isfinite(sigma2) or sigma2 <= 0:
            sigma2 = 1.0

    d2_xx = pairwise_sq_dists(x, x)
    d2_yy = pairwise_sq_dists(y, y)
    d2_xy = pairwise_sq_dists(x, y)

    Kxx = np.exp(-d2_xx / (2.0 * sigma2))
    Kyy = np.exp(-d2_yy / (2.0 * sigma2))
    Kxy = np.exp(-d2_xy / (2.0 * sigma2))

    n = x.shape[0]
    m = y.shape[0]
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


def sliced_w2_distance(
    x: np.ndarray,
    y: np.ndarray,
    directions: np.ndarray,
) -> float:
    proj_x = x @ directions.T
    proj_y = y @ directions.T

    n_q = min(x.shape[0], y.shape[0])
    if n_q < 2:
        return float("nan")

    q = (np.arange(n_q, dtype=np.float64) + 0.5) / n_q
    qx = np.quantile(proj_x, q, axis=0)
    qy = np.quantile(proj_y, q, axis=0)
    w2_sq_dir = ((qx - qy) ** 2).mean(axis=0)
    return float(np.sqrt(w2_sq_dir.mean()))


def write_metric_csv(detail_df: pd.DataFrame, metric_col: str, avg_out: Path):
    grouped = (
        detail_df.groupby(["pair", "k", "M1"], dropna=False)[metric_col]
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
        default=10000,
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
    args = parser.parse_args()

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

    total_ckpt = sum(len(ckpts) for _, ckpts in jobs)
    rows = []
    pbar = tqdm(total=total_ckpt, desc="Evaluating checkpoints", unit="ckpt")
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
                    mmd_val = mmd_rbf_unbiased(mapped, target)
                    sw2_val = sliced_w2_distance(mapped, target, directions=directions)

                    rows.append(
                        {
                            "pair": pair_name,
                            "k": k_val,
                            "M1": m1_label,
                            "model_num": model_num,
                            "epoch": epoch,
                            "checkpoint": str(ckpt_path),
                            "mmd_rbf": mmd_val,
                            "sw2": sw2_val,
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
        metric_col="mmd_rbf",
        avg_out=Path(args.output_mmd_csv),
    )
    write_metric_csv(
        detail_df=detail_df,
        metric_col="sw2",
        avg_out=Path(args.output_sw2_csv),
    )

    print(f"Saved MMD average CSV to: {args.output_mmd_csv}")
    print(f"Saved SW2 average CSV to: {args.output_sw2_csv}")


if __name__ == "__main__":
    main()
