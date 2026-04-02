import os, time, itertools, argparse, math
from multiprocessing import Pool

import torch
import torch.optim as optim
from torch.utils.data import DataLoader

from conjugate_optimization import *
from dataset import *
from network import *

def _is_infinite_m1(m1):
    return math.isinf(float(m1))


def _format_m1_for_path(m1):
    m1 = float(m1)
    if m1.is_integer():
        return str(int(m1))
    return f"{m1:.12g}"


def _parse_m1_values(arg: str):
    values = []
    for token in arg.split(","):
        t = token.strip().lower()
        if not t:
            continue
        if t in {"inf", "infty", "infinity"}:
            values.append(float("inf"))
        else:
            values.append(float(token))
    if not values:
        raise ValueError("--m1_values must include at least one value.")

    # Preserve order but deduplicate.
    out = []
    for v in values:
        if not any((math.isinf(v) and math.isinf(u)) or (not math.isinf(v) and v == u) for u in out):
            out.append(v)
    return out


def _make_phi(model, m1):
    if _is_infinite_m1(m1):
        return model
    return lambda x: model.forward_truncated(x, m1)


def train_model_epoch_proj_slow(model, num_epochs_cvx_conjugate, dataloader, optimizer, M1=0):
    M2 = dataloader.dataset.M2
    phi = _make_phi(model, M1)

    for x_in, y_in, _ in dataloader:
        loss = phi(x_in).mean() + cvx_conjugate_proj_slow(y_in, phi, num_epochs_cvx_conjugate, M2).mean()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()     
        clip_parameters(model) 


def train_model_epoch_noproj_slow(model, num_epochs_cvx_conjugate, dataloader, optimizer, M1=0):
    phi = _make_phi(model, M1)

    for x_in, y_in, _ in dataloader:
        # Forward pass
        loss = phi(x_in).mean() + cvx_conjugate_slow(y_in, phi, num_epochs_cvx_conjugate).mean()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        clip_parameters(model)


def get_data_loader(sample_size, measure_P, transform_method, df, input_size, scale_k, batch_size, device):
    
    x, y = generate_raw_data(sample_size, measure_P, transform_method, df, input_size)

    dataset = CustomDataset(x, y, scale_k, device)
    dataloader = DataLoader(dataset, batch_size, shuffle=True)

    return dataloader


def get_model_and_optim(input_size, hidden_size, act, learning_rate, device):

    output_size = 1
    model = ICNN(input_size, hidden_size, output_size, act)
    clip_parameters(model)

    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    model.train()

    return model, optimizer


def train_model(args):

    torch.set_num_threads(1)
    model_num, batch_size, input_size, hidden_size, act, learning_rate, sample_size, transform_method, measure_P, df, scale_k, M1, num_epochs, num_epochs_cvx_conjugate, path, device = args
    
    localtime = time.asctime( time.localtime(time.time()) )
    M1_label = "inf" if _is_infinite_m1(M1) else _format_m1_for_path(M1)
    print(f"start training model {model_num} with d={input_size}, n={sample_size}, {measure_P}, {transform_method}, k={scale_k}, M1={M1_label} at:" + localtime)

    dataloader = get_data_loader(sample_size, measure_P, transform_method, df, input_size, scale_k, batch_size, device)
    model, optimizer = get_model_and_optim(input_size, hidden_size, act, learning_rate, device)

    #x, y = generate_raw_data(sample_size, measure_P, transform_method, df, input_size)
    if scale_k == -1:
        for _ in range(num_epochs):
            train_model_epoch_noproj_slow(model, num_epochs_cvx_conjugate, dataloader, optimizer, M1=M1)
    else: 
        for _ in range(num_epochs):
            train_model_epoch_proj_slow(model, num_epochs_cvx_conjugate, dataloader, optimizer, M1=M1)


    model_path = f"{path}model_{model_num}.pth"
    torch.save(model.state_dict(), model_path)
    print(f"store trained model {model_num} at: " + model_path)


def train_each_model(d, n, measure, transform, k, m1, input_index):
    # Hyperparameters
    input_size = d
    hidden_size = 16

    num_epochs = 500
    num_epochs_cvx_conjugate = 500

    learning_rate = 0.001

    sample_size = n
    batch_size = 50
    scale_k = k
    M1 = m1

    measure_P = measure # "normal", "t"
    df = 6
    activation = "softplus_scaled"

    transform_method = transform # "CDF", "linear", "quadratic"
    ##number_of_worker = args.worker

    device = "cpu"

    path_base = f"../simulation_results/d={input_size}/{measure_P}_{transform_method}_n_{sample_size}_k_{scale_k}"
    if _is_infinite_m1(M1):
        path = f"{path_base}/"
    else:
        path = f"{path_base}_M1_{_format_m1_for_path(M1)}/"
    #print(path)

    if not os.path.exists(path):
        os.makedirs(path)

    arguments = (input_index, batch_size, input_size, hidden_size, activation, learning_rate, 
                sample_size, transform_method, measure_P, df, 
                scale_k, M1, num_epochs, num_epochs_cvx_conjugate, path, device
                )
    
    model_path_i = f"{path}model_{input_index}.pth"
    if not os.path.exists(model_path_i):
        #print(model_path_i)
        
        train_model(arguments)

        
def build_combos(m1_values):
    dimensions = [20, 10]
    sample_sizes = [1000, 500, 300, 100]
    measures = ["t", "normal"]
    transforms = ["CDF", "piecewise_linear", "quadratic"]
    scale_ks = [-1.0, 1.0, 2.0]
    model_numbers = range(100)

    return [
        {"d": d,
         "n": n,
         "measure": m,
         "transform": t,
         "k": k,
         "m1": m1,
         "input_index": idx}
        for d, n, m, t, k, m1, idx in itertools.product(
            dimensions, sample_sizes, measures, transforms, scale_ks, m1_values, model_numbers
        )
    ]

def worker(hp):
    return train_each_model(**hp)

if __name__=="__main__":
    parser = argparse.ArgumentParser(description="Run OT simulation grid.")
    parser.add_argument(
        "--threads",
        type=int,
        default=5,
        help="Number of worker processes for multiprocessing Pool (default: 5).",
    )
    parser.add_argument(
        "--m1_values",
        type=str,
        default="inf",
        help="Comma-separated M1 values, e.g. 'inf' or 'inf,10,20'.",
    )
    args = parser.parse_args()

    if args.threads < 1:
        raise ValueError("--threads must be >= 1")
    m1_values = _parse_m1_values(args.m1_values)
    combos = build_combos(m1_values)

    with Pool(args.threads) as pool:
        async_results = [
            pool.apply_async(worker, (hp,))
            for hp in combos
        ]
        for ar in async_results:
            ar.wait()
