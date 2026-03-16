import torch
import torch.optim as optim

def inner_product(a, b):
    """Batch inner product ⟨a_i, b_i⟩ over arbitrary trailing dims."""
    return (a * b).reshape(a.size(0), -1).sum(dim=1)

# Function to compute the convex conjugate phi_star(y) using optimization
def cvx_conjugate_slow(y, phi, inner_epoch):
    x = torch.zeros_like(y, requires_grad=True)  # Initialize x as zero tensor with gradients
    optimizer = optim.SGD([x], lr=0.001) 

    # Closure to compute the objective function and its gradient
    def closure():
        optimizer.zero_grad()
        loss = (phi(x) - inner_product(x, y)).sum()
        loss.backward()
        return loss
    
    # Run the optimizer
    for _ in range(inner_epoch):
        optimizer.step(closure)
    
    # Return the optimal value of the convex conjugate
    phi_star_y = (inner_product(x, y) - phi(x))
    
    return phi_star_y


def cvx_conjugate_proj_slow(y, phi, inner_epoch, Mn):
    x = torch.zeros_like(y, requires_grad=True)  # Initialize x as zero tensor with gradients
    optimizer = optim.SGD([x], lr=0.001) 

    # Closure to compute the objective function and its gradient
    def closure():
        optimizer.zero_grad()
        loss = (phi(x) - inner_product(x, y)).sum()
        loss.backward(retain_graph=True)  # Retain the computation graph
        return loss
    
    # Run the optimizer
    for _ in range(inner_epoch):
        optimizer.step(closure)

        # project them to the ball B(0, Mn)
        norms = torch.norm(x, dim=1)
        scaling_factors = torch.clamp(Mn / norms, max=1.0)
        x.data = x * scaling_factors.unsqueeze(1)

    # Return the optimal value of the convex conjugate
    phi_star_y = (inner_product(x, y) - phi(x))
    
    return phi_star_y