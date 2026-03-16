# ---------------------------------------------------------------------
# NOTICE:
# This file includes an adaptation of the Input Convex Neural Network (ICNN) originally
# implemented in TensorFlow and licensed under the Apache License 2.0 by the original authors:
# Brandon Amos, Lei Xu and Zico Kolter.
# The original TensorFlow code can be found here: https://github.com/locuslab/icnn.
#
# Modifications have been made to convert the code to PyTorch. These modifications are 
# licensed under the Apache License 2.0. A copy of this license is included in this 
# repository as `Apache_LICENSE`, or you can access it here: 
# http://www.apache.org/licenses/LICENSE-2.0.
# ---------------------------------------------------------------------

import torch
import torch.nn as nn

class CenteredSoftplus(nn.Module):
    """
    sigma(x) = softplus(x) - softplus(0)
    """
    def __init__(self):
        super().__init__()
        self.softplus = nn.Softplus()
        # Precompute softplus(0) as float32
        self.c0 = float(self.softplus(torch.zeros(1)))

    def forward(self, x):
        return self.softplus(x) - self.c0

class ScaledSoftplus(nn.Module):
    def __init__(self, scale: float = 1.05):
        super().__init__()
        self.softplus = nn.Softplus()
        self.c0 = float(self.softplus(torch.zeros(1)))
        self.scale = float(torch.tensor(float(scale)))

    def forward(self, x):
        return self.scale * (self.softplus(x) - self.c0)
    
def make_activation_module(name: str) -> nn.Module:
    name = name.lower()
    if name == "softplus":
        return nn.Softplus()
    if name == "softplus_centered":
        return CenteredSoftplus()
    if name == "softplus_scaled":
        return ScaledSoftplus()
    
    raise ValueError("Unsupported activation.")


class ICNN(nn.Module):
    def __init__(
        self,
        input_size,
        hidden_size,
        output_size,
        activation,
        num_hidden_layers=None,
    ):
        """
        Backward-compatible ICNN.

        - If num_hidden_layers is None: uses the legacy 3-layer structure
          (fc1 -> fc2 -> fc3) so existing code/state_dict keys are unchanged.
        - If num_hidden_layers is an int >= 1: builds a flexible-depth ICNN.
        """
        super(ICNN, self).__init__()
        self.activation = make_activation_module(activation)

        self.use_legacy_layout = (num_hidden_layers is None)
        if self.use_legacy_layout:
            # Legacy parameter names preserved for backward compatibility.
            self.fc1 = nn.Linear(input_size, hidden_size)
            self.fc2_z = nn.Linear(hidden_size, hidden_size, bias=False)
            self.fc2_x = nn.Linear(input_size, hidden_size)
            self.fc3_z = nn.Linear(hidden_size, output_size, bias=False)
            self.fc3_x = nn.Linear(input_size, output_size)
        else:
            if int(num_hidden_layers) < 1:
                raise ValueError("num_hidden_layers must be >= 1 when specified.")
            self.num_hidden_layers = int(num_hidden_layers)

            self.fc1 = nn.Linear(input_size, hidden_size)

            n_middle = self.num_hidden_layers - 1
            self.z_layers = nn.ModuleList(
                [nn.Linear(hidden_size, hidden_size, bias=False) for _ in range(n_middle)]
            )
            self.x_layers = nn.ModuleList(
                [nn.Linear(input_size, hidden_size) for _ in range(n_middle)]
            )

            self.out_z = nn.Linear(hidden_size, output_size, bias=False)
            self.out_x = nn.Linear(input_size, output_size)

    def forward(self, x):
        if self.use_legacy_layout:
            z = self.fc1(x)
            z = self.activation(z)

            z = self.fc2_z(z) + self.fc2_x(x)
            z = self.activation(z)

            z = self.fc3_z(z) + self.fc3_x(x)
            return z

        z = self.fc1(x)
        z = self.activation(z)
        for z_layer, x_layer in zip(self.z_layers, self.x_layers):
            z = z_layer(z) + x_layer(x)
            z = self.activation(z)
        z = self.out_z(z) + self.out_x(x)
        return z
    
    @torch.no_grad()
    def _phi0_no_grad(self, x_like: torch.Tensor) -> torch.Tensor:
        """Compute φ(0) with no grad; returns a tensor broadcastable to forward(x)."""
        zero_in = torch.zeros_like(x_like)
        phi0 = self.forward(zero_in)             # shape: (B, output_size)
        return phi0

    def forward_truncated(self, x: torch.Tensor, M1: float, detach_phi0: bool = True) -> torch.Tensor:
        """
        Evaluate φ(x) and clamp it into [φ(0)-M1, φ(0)+M1] elementwise.

        Args:
            x:  (B, input_size)
            M1: positive scalar
            detach_phi0: if True (default), φ(0) is treated as a constant (no grad).
                         If False, φ(0) is computed with grad (note: torch.clamp
                         does not propagate gradients to its bounds in any case).

        Returns:
            z_trunc: truncated output with same shape as forward(x)
        """
        z = self.forward(x)  # φ(x), shape (B, output_size)

        if detach_phi0:
            # Efficient: compute φ(0) without building a graph
            phi0 = self._phi0_no_grad(x)
        else:
            # Graph-enabled φ(0); clamp won't backprop to bounds, but we expose the option
            zero_in = torch.zeros_like(x, requires_grad=True)
            phi0 = self.forward(zero_in)

        M1_t = torch.as_tensor(M1, dtype=z.dtype, device=z.device)
        lower = phi0 - M1_t
        upper = phi0 + M1_t

        # Elementwise truncation
        z_trunc = torch.clamp(z, min=lower, max=upper)
        return z_trunc
    

def clip_parameters(model):
    # Ensure non-negative weights for specific layers
    for name, param in model.named_parameters():
        if (
            "fc2_z" in name
            or "fc3_z" in name
            or "z_layers" in name
            or "out_z" in name
        ):
            param.data = torch.clamp(param.data, min=0.0)


def compute_gradients(model, x: torch.Tensor) -> torch.Tensor:
    x_ = x.clone().detach().requires_grad_(True)
    model.eval()
    out = model(x_)
    model.zero_grad(set_to_none=True)
    out.backward(torch.ones_like(out))
    return x_.grad.detach()


def load_model(input_size, hidden_size, model_path, num_hidden_layers=None):
    
    output_size = 1
    
    model = ICNN(
        input_size,
        hidden_size,
        output_size,
        activation="softplus_scaled",
        num_hidden_layers=num_hidden_layers,
    )

    model.load_state_dict(torch.load(model_path))
    
    return model
