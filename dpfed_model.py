import torch
import torch.nn as nn
import numpy as np

def unflatten_block(block, index, weights, device):
    """
    Unflatten the given block into the model at specified index and transfer to the device
    """
    block_state_dict = block.state_dict()
    for key, value in block_state_dict.items():
        param = value.cpu().detach().numpy()
        size = param.shape
        param = param.flatten()
        num_elements = len(param)
        weight = weights[index:index + num_elements]
        index += num_elements
        np_arr = np.array(weight).reshape(size)
        block_state_dict[key] = torch.tensor(np_arr).to(device)

    block.load_state_dict(block_state_dict)
    return index


class DpFedModel(torch.nn.Module):
    def __init__(self, model, device):
        super(DpFedModel, self).__init__()
        self.model = model.to(device)
        self.device = device

    def forward(self, X):
        """
        Calculate loss using the model
        """
        loss = self.layer(X)
        return loss

    def flatten(self):
        """
        Flatten the model into a linear array on CPU
        """
        all_params = np.array([])

        for key, value in self.layer.state_dict().items():
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
        index = 0
        index = unflatten_block(self.layer, 0, weights, self.device)
