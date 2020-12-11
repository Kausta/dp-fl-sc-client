import numpy as np
import torch
from fl_dp.models import MnistMLP


class FedServer:
    clients: dict = None
    device = None
    init_model = None

    def __init__(self):
        print("FedServer: Cuda:", torch.cuda.is_available())
        if torch.cuda.is_available():
            dev = "cuda:0"
        else:
            dev = "cpu"
        print("Using device", dev)
        self.device = torch.device(dev)
        self.init_model = MnistMLP(self.device)

    def set_clients(self, clients):
        print("FedServer: Saved clients.")
        self.clients = clients

    def get_public_parameters(self):
        return {"system_size": len(self.clients),
                "group_desc": 14,
                "lr": 0.01,
                "S": 1,
                "epsilon": 1,
                "batch_size": 100,
                "local_epochs": 100,
                "data_dir": "data/mnist/",
                "target_acc": 0.95,
                "rounds": 1}

    def get_init_model_flattened(self):
        return self.init_model.flatten()

    def aggregate(self, updates, weights):
        total_weight = np.sum(weights)
        update = np.zeros_like(updates[0])
        for weight, local_update in zip(weights, updates):
            update += weight * local_update
        update /= total_weight
        return update
