import torch
import torch.nn as nn

from fl_dp.dpfed_model import DpFedModel


class MnistMLP(DpFedModel):
    def __init__(self, device=torch.device("cpu")):
        layer = nn.Sequential(
            nn.Linear(784, 200),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(200, 200),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(200, 10),
            nn.LogSoftmax(dim=1)
        )
        super(MnistMLP, self).__init__(layer, device)

    def forward(self, X):
        loss = self.model(X.view(X.shape[0], -1))
        return loss
