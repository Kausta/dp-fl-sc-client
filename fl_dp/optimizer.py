import torch
from torch.optim import Optimizer


# Based on torch.optim.SGD, simplified to match DP-FedAvg, has no changing state
class SimpleSGDOptimizer(Optimizer):
    def __init__(self, params, lr):
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        defaults = dict(lr=lr)
        super(SimpleSGDOptimizer, self).__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue

                d_p = p.grad
                p.add_(d_p, alpha=-group['lr'])

        return loss
