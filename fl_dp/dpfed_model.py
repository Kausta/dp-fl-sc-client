import numpy as np
import torch
import torch.nn as nn


class DpFedModel(torch.nn.Module):
    def __init__(self, model, device):
        super(DpFedModel, self).__init__()
        self.model = model.to(device)
        self.device = device

    def forward(self, X):
        """
        Calculate loss using the model
        """
        loss = self.model(X)
        return loss

    def flatten(self):
        """
        Flatten the model into a linear array on CPU
        """
        all_params = np.array([])

        for key, value in self.model.state_dict().items():
            param = value.cpu().detach().numpy().flatten()
            all_params = np.append(all_params, param)

        return all_params

    @torch.no_grad()
    def unflatten(self, weights):
        """
        Load the weights from a linear array on CPU to the actual model on the device
        :param weights:
        :return:
        """
        block_state_dict = self.model.state_dict()
        index = 0
        for key, value in block_state_dict.items():
            param = value.cpu().detach().numpy()
            size = param.shape
            param = param.flatten()
            num_elements = len(param)
            weight = weights[index:index + num_elements]
            index += num_elements
            np_arr = np.array(weight).reshape(size)
            block_state_dict[key] = torch.tensor(np_arr).to(self.device)

        self.model.load_state_dict(block_state_dict)

    def get_layer_tensors(self, cpu=True, copy=False):
        tensors = []
        for parameter in self.parameters():
            param = parameter.data
            if copy:
                param = param.detach().clone()
            if cpu:
                param = param.cpu()
            tensors.append(param)
        return tensors

    @torch.no_grad()
    def set_layer_tensors(self, tensors):
        for tensor, parameter in zip(tensors, self.parameters()):
            parameter.data = tensor.to(self.device)
