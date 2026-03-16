import argparse
import itertools
import json
import math
import time
from multiprocessing import get_context
from multiprocessing import current_process
from pathlib import Path

import numpy as np
import torch
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR, StepLR
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from conjugate_optimization import cvx_conjugate_proj_slow, cvx_conjugate_slow
from network import ICNN, clip_parameters


def parse_float_list(raw: str):
    values = []
    for token in raw.split(","):
        token = token.strip().lower()
        if not token:
            continue
        if token in {"inf", "+inf", "infty", "infinity"}:
            values.append(float("inf"))
        else:
            values.append(float(token))
    return values


def parse_k_list(raw: str):
    return [float(x.strip()) for x in raw.split(",") if x.strip()]


def get_default_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    mps_ok = hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
    if mps_ok:
        return "mps"
    return "cpu"


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


def format_float_tag(x: float) -> str:
    if math.isinf(x):
        return "inf"
    return f"{x:g}".replace("-", "neg_").replace(".", "p")


def apply_phi(model: ICNN, x: torch.Tensor, m1: float) -> torch.Tensor:
    if math.isinf(m1):
        return model(x)
    return model.forward_truncated(x, M1=m1)


def train_model_epoch(
    model: ICNN,
    num_epochs_cvx_conjugate: int,
    bull_loader: DataLoader,
    bear_loader: DataLoader,
    steps_per_epoch: int,
    M2: float,
    optimizer: optim.Optimizer,
    scale_k: float,
    m1: float,
) -> float:
    epoch_loss = 0.0
    use_projection = (scale_k != -1.0)

    bull_iter = iter(bull_loader)
    bear_iter = iter(bear_loader)

    for _ in range(steps_per_epoch):
        try:
            x_in = next(bull_iter)[0]
        except StopIteration:
            bull_iter = iter(bull_loader)
            x_in = next(bull_iter)[0]

        try:
            y_in = next(bear_iter)[0]
        except StopIteration:
            bear_iter = iter(bear_loader)
            y_in = next(bear_iter)[0]

        phi_fn = lambda z: apply_phi(model, z, m1)
        if use_projection:
            loss = phi_fn(x_in).mean() + cvx_conjugate_proj_slow(
                y_in, phi_fn, num_epochs_cvx_conjugate, M2
            ).mean()
        else:
            loss = phi_fn(x_in).mean() + cvx_conjugate_slow(
                y_in, phi_fn, num_epochs_cvx_conjugate
            ).mean()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        clip_parameters(model)

        epoch_loss += float(loss.item())

    if steps_per_epoch == 0:
        return float("nan")
    return epoch_loss / steps_per_epoch


def get_model_and_optim(
    input_size: int,
    hidden_size: int,
    num_hidden_layers,
    activation: str,
    learning_rate: float,
    device: str,
    scheduler_name: str,
    num_epochs: int,
):
    output_size = 1
    model = ICNN(
        input_size,
        hidden_size,
        output_size,
        activation,
        num_hidden_layers=num_hidden_layers,
    )
    clip_parameters(model)
    model = model.to(device)

    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    if scheduler_name == "none":
        scheduler = None
    elif scheduler_name == "cosine":
        scheduler = CosineAnnealingLR(optimizer, T_max=max(num_epochs, 1))
    elif scheduler_name == "step":
        step_size = max(num_epochs // 3, 1)
        scheduler = StepLR(optimizer, step_size=step_size, gamma=0.5)
    else:
        raise ValueError(f"Unsupported scheduler: {scheduler_name}")

    model.train()
    return model, optimizer, scheduler


def load_train_arrays(data_dir: Path):
    x_train = np.load(data_dir / "train_bull.npy")
    y_train = np.load(data_dir / "train_bear.npy")
    if x_train.ndim != 2 or y_train.ndim != 2:
        raise ValueError("Expected 2D arrays for train_bull.npy and train_bear.npy")
    if x_train.shape[1] != y_train.shape[1]:
        raise ValueError("train_bull and train_bear must have the same feature dimension")
    return x_train.astype(np.float32), y_train.astype(np.float32)


def get_data_loaders_from_finance(
    train_bull: np.ndarray,
    train_bear: np.ndarray,
    scale_k: float,
    batch_size: int,
    device: str,
    seed: int,
):
    n_bull = train_bull.shape[0]
    n_bear = train_bear.shape[0]
    if n_bull == 0 or n_bear == 0:
        raise ValueError("Bull and bear train arrays must be non-empty")

    x = torch.from_numpy(train_bull).to(device, non_blocking=True)
    y = torch.from_numpy(train_bear).to(device, non_blocking=True)

    M2 = scale_k * torch.max(torch.norm(x, dim=1)).item()
    steps_per_epoch = int(math.ceil(max(n_bull, n_bear) / batch_size))

    bull_gen = torch.Generator()
    bull_gen.manual_seed(seed)
    bear_gen = torch.Generator()
    bear_gen.manual_seed(seed + 1)

    bull_loader = DataLoader(
        TensorDataset(x),
        batch_size=batch_size,
        shuffle=True,
        drop_last=False,
        generator=bull_gen,
    )
    bear_loader = DataLoader(
        TensorDataset(y),
        batch_size=batch_size,
        shuffle=True,
        drop_last=False,
        generator=bear_gen,
    )

    return bull_loader, bear_loader, {
        "n_bull": n_bull,
        "n_bear": n_bear,
        "steps_per_epoch": steps_per_epoch,
        "M2": M2,
        "epoch_def": "ceil(max(n_bull,n_bear)/batch_size)",
    }


def maybe_save_checkpoint(
    model: ICNN,
    optimizer: optim.Optimizer,
    scheduler,
    epoch: int,
    num_epochs: int,
    save_every: int,
    model_dir: Path,
):
    if epoch % save_every != 0 and epoch != num_epochs:
        return None

    ckpt_path = model_dir / f"epoch_{epoch:04d}.pth"
    payload = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
    }
    if scheduler is not None:
        payload["scheduler_state_dict"] = scheduler.state_dict()
    torch.save(payload, ckpt_path)
    return ckpt_path


def get_worker_position() -> int:
    identity = current_process()._identity
    if not identity:
        return 0
    return max(0, int(identity[0]) - 1)


def train_one_config(args_tuple):
    torch.set_num_threads(1)

    (
        model_num,
        k,
        m1,
        input_size,
        hidden_size,
        num_hidden_layers,
        activation,
        learning_rate,
        batch_size,
        num_epochs,
        inner_epochs,
        save_every,
        scheduler_name,
        seed,
        data_dir,
        output_root,
        device,
    ) = args_tuple

    m1_tag = format_float_tag(m1)
    k_tag = format_float_tag(k)
    model_dir = output_root / f"k_{k_tag}" / f"M1_{m1_tag}" / f"model_{model_num:03d}"
    model_dir.mkdir(parents=True, exist_ok=True)
    final_ckpt = model_dir / f"epoch_{num_epochs:04d}.pth"
    if final_ckpt.exists():
        # print(f"[skip] {final_ckpt} already exists")
        return

    # localtime = time.asctime(time.localtime(time.time()))
    # print(
    #     f"start model={model_num} k={k} M1={m1_tag} "
    #     f"(epochs={num_epochs}, inner={inner_epochs}) at {localtime}"
    # )

    train_bull, train_bear = load_train_arrays(data_dir)
    bull_loader, bear_loader, data_stats = get_data_loaders_from_finance(
        train_bull=train_bull,
        train_bear=train_bear,
        scale_k=k,
        batch_size=batch_size,
        device=device,
        seed=seed + model_num,
    )

    model, optimizer, scheduler = get_model_and_optim(
        input_size=input_size,
        hidden_size=hidden_size,
        num_hidden_layers=num_hidden_layers,
        activation=activation,
        learning_rate=learning_rate,
        device=device,
        scheduler_name=scheduler_name,
        num_epochs=num_epochs,
    )

    run_config = {
        "model_num": model_num,
        "k": k,
        "M1": "inf" if math.isinf(m1) else m1,
        "input_size": input_size,
        "hidden_size": hidden_size,
        "num_hidden_layers": num_hidden_layers,
        "activation": activation,
        "learning_rate": learning_rate,
        "batch_size": batch_size,
        "num_epochs": num_epochs,
        "inner_epochs": inner_epochs,
        "save_every": save_every,
        "scheduler": scheduler_name,
        "seed": seed,
        "device": device,
        "data_stats": data_stats,
    }
    (model_dir / "config.json").write_text(json.dumps(run_config, indent=2))

    log_path = model_dir / "train_log.csv"
    with log_path.open("w") as f:
        f.write("epoch,loss,lr\n")
        progress = tqdm(
            total=num_epochs,
            desc=f"w{get_worker_position()} k={k:g} M1={m1_tag} m={model_num}",
            position=get_worker_position(),
            dynamic_ncols=True,
            leave=True,
        )
        try:
            for epoch in range(1, num_epochs + 1):
                loss = train_model_epoch(
                    model=model,
                    num_epochs_cvx_conjugate=inner_epochs,
                    bull_loader=bull_loader,
                    bear_loader=bear_loader,
                    steps_per_epoch=data_stats["steps_per_epoch"],
                    M2=data_stats["M2"],
                    optimizer=optimizer,
                    scale_k=k,
                    m1=m1,
                )
                if scheduler is not None:
                    scheduler.step()
                lr = optimizer.param_groups[0]["lr"]
                f.write(f"{epoch},{loss:.8f},{lr:.10f}\n")
                _ = maybe_save_checkpoint(
                    model=model,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    epoch=epoch,
                    num_epochs=num_epochs,
                    save_every=save_every,
                    model_dir=model_dir,
                )
                progress.set_postfix({"loss": f"{loss:.4f}", "lr": f"{lr:.2e}"})
                progress.update(1)
        finally:
            progress.close()

    # print(f"finished model={model_num} k={k} M1={m1_tag} -> {final_ckpt}")


def build_combos(num_models: int, k_values, m1_values):
    combos = []
    for k in k_values:
        if k == -1.0:
            m1_grid = [float("inf")]
        else:
            m1_grid = m1_values
        for m1, idx in itertools.product(m1_grid, range(num_models)):
            combos.append({"k": float(k), "m1": float(m1), "model_num": int(idx)})
    return combos


def worker(job):
    return train_one_config(job)


def main():
    parser = argparse.ArgumentParser(description="Train OT maps from bull to bear data.")
    parser.add_argument("--threads", type=int, default=2, help="Worker processes.")
    parser.add_argument(
        "--device",
        type=str,
        default=get_default_device(),
        help="Training device.",
    )
    parser.add_argument("--num-models", type=int, default=1, help="Model seeds per config.")
    parser.add_argument("--num-epochs", type=int, default=500, help="Outer epochs.")
    parser.add_argument("--inner-epochs", type=int, default=500, help="Conjugate epochs.")
    parser.add_argument("--batch-size", type=int, default=128, help="Batch size.")
    parser.add_argument("--hidden-size", type=int, default=32, help="ICNN hidden width.")
    parser.add_argument(
        "--num-hidden-layers",
        type=int,
        default=5,
        help="Number of hidden layers for ICNN. If not set, uses legacy architecture.",
    )
    parser.add_argument("--learning-rate", type=float, default=1e-3, help="Adam learning rate.")
    parser.add_argument(
        "--activation",
        type=str,
        default="softplus_scaled",
        choices=["softplus", "softplus_centered", "softplus_scaled"],
        help="ICNN activation.",
    )
    parser.add_argument(
        "--scheduler",
        type=str,
        default="cosine",
        choices=["none", "cosine", "step"],
        help="Learning-rate scheduler.",
    )
    parser.add_argument("--save-every", type=int, default=10, help="Checkpoint interval.")
    parser.add_argument("--seed", type=int, default=2026, help="Random seed base.")
    parser.add_argument(
        "--k-values",
        type=str,
        default="-1, 1, 2",
        help="Comma-separated k values.",
    )
    parser.add_argument(
        "--m1-values",
        type=str,
        default="inf, 32",
        help="Comma-separated M1 values for k != -1.",
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default="data_finance/processed/npy",
        help="Directory containing train_bull.npy and train_bear.npy.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="model_finance",
        help="Directory for storing trained model checkpoints.",
    )
    args = parser.parse_args()

    if args.threads < 1:
        raise ValueError("--threads must be >= 1")
    if args.save_every < 1:
        raise ValueError("--save-every must be >= 1")
    if args.num_models < 1:
        raise ValueError("--num-models must be >= 1")

    args.device = validate_device(args.device)

    data_dir = Path(args.data_dir)
    output_root = Path(args.output_dir)
    output_root.mkdir(parents=True, exist_ok=True)

    train_bull_path = data_dir / "train_bull.npy"
    train_bear_path = data_dir / "train_bear.npy"
    if not train_bull_path.exists() or not train_bear_path.exists():
        raise FileNotFoundError(
            f"Missing train arrays in {data_dir}. "
            "Expected train_bull.npy and train_bear.npy."
        )

    train_bull, train_bear = load_train_arrays(data_dir)
    input_size = int(train_bull.shape[1])
    if input_size != int(train_bear.shape[1]):
        raise ValueError("Feature dimension mismatch between train_bull and train_bear")

    k_values = parse_k_list(args.k_values)
    m1_values = parse_float_list(args.m1_values)
    if not any(math.isinf(v) for v in m1_values):
        m1_values = [float("inf")] + m1_values
    m1_values = sorted(set(m1_values), key=lambda x: (0 if math.isinf(x) else 1, x))

    combos = build_combos(
        num_models=args.num_models,
        k_values=k_values,
        m1_values=m1_values,
    )
    # print(f"total training jobs: {len(combos)}")

    jobs = []
    for hp in combos:
        jobs.append(
            (
                hp["model_num"],
                hp["k"],
                hp["m1"],
                input_size,
                args.hidden_size,
                args.num_hidden_layers,
                args.activation,
                args.learning_rate,
                args.batch_size,
                args.num_epochs,
                args.inner_epochs,
                args.save_every,
                args.scheduler,
                args.seed,
                data_dir,
                output_root,
                args.device,
            )
        )

    with get_context("spawn").Pool(args.threads) as pool:
        async_results = [pool.apply_async(worker, (job,)) for job in jobs]
        for ar in async_results:
            ar.get()


if __name__ == "__main__":
    main()
