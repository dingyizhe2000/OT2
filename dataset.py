import torch
from torch.utils.data import Dataset

import scipy.stats as stats

class CustomDataset(Dataset):
    def __init__(self, x, y, k, device):
        self.x = x.to(device, non_blocking=True)
        self.y = y.to(device, non_blocking=True)

        self.M2 = k * torch.max(torch.norm(self.x, dim=1)).item()

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx], idx
    

class sign_square_class:
    def __call__(self, x):
        transformed_x = torch.where(x > 0, x**2, -x**2)
        return transformed_x

sign_square = sign_square_class()
def generate_raw_data(sample_size, measure_P, transform_method, df, input_size):
    
    t_distribution = torch.distributions.StudentT(df)
    
    if measure_P=="normal":
        x = torch.randn(sample_size, input_size)
        z = torch.randn(sample_size, input_size)
        y = torch.tensor(stats.norm.cdf(z), dtype=torch.float32)
    elif measure_P=="t":
        x = t_distribution.sample((sample_size, input_size))
        z = t_distribution.sample((sample_size, input_size))
        y = torch.tensor(stats.t.cdf(z, df), dtype=torch.float32)
    
    if transform_method == "piecewise_linear":
        # y(z) =
        #   z                                      if |z| <= 1
        #   sgn(z) * (0.5 * (|z| - 1) + 1)        if 1 < |z| <= 2
        #   sgn(z) * (2 * (|z| - 2) + 1.5)        if |z| > 2

        abs_z  = z.abs()
        sign_z = z.sign()

        middle = sign_z * (0.5 * (abs_z - 1.0) + 1.0)     # slope 0.5 region
        outer  = sign_z * (2.0 * (abs_z - 2.0) + 1.5)     # slope 2 region

        y = torch.where(
            abs_z <= 1.0,
            z,
            torch.where(abs_z <= 2.0, middle, outer)
        )

    elif transform_method=="quadratic":
        y = sign_square(z)
        
    return x, y