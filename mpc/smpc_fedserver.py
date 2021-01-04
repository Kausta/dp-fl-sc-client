import numpy as np
import torch
from fl_dp.models import MnistMLP


class FedServer:
    clients: dict = None
    device = None
    init_model = None

    def __init__(self):
        self.device = torch.device("cpu")
        self.init_model = MnistMLP(self.device)

    def set_clients(self, clients):
        print("FedServer: Saved clients.")
        self.clients = clients

    def get_public_parameters(self):
        return {"system_size": len(self.clients),
                "group_desc": 14,
                'data_dir': "data/mnist/",
                'key_path': 'test_key.pem',
                'factor_exp': 16,
                'lr': 0.01,
                'S': 1,
                'epsilon': 1,
                'batch_size': 100,
                'local_epochs': 5,
                'target_acc': 0.90,
                'q': 1}

    def get_init_model_flattened(self):
        return self.init_model.flatten()

    def aggregate(self, updates, weights):
        total_weight = np.sum(weights)
        update = np.zeros_like(updates[0])
        for weight, local_update in zip(weights, updates):
            update += weight * local_update
        update /= total_weight
        return update
