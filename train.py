import abc

import torch
import numpy as np

from dpfed_model import DpFedModel
from optimizer import SimpleSGDOptimizer


def projection(vec, S):
    return vec * np.min(1, S / np.linalg.norm(vec, ord=2))


def flat_clip(vec, S):
    return projection(vec, S)


class DpFedStep:
    def __init__(self, model: DpFedModel, loader, lr, S):
        self.model = model
        self.loader = loader
        self.optimizer = SimpleSGDOptimizer(self.model.parameters(), lr)

        self.lr = lr
        self.S = S

    def _train_epoch(self, initial_params):
        criterion = torch.nn.NLLLoss()
        self.model.train()

        for i, (x_batch, y_batch) in enumerate(self.loader):
            x_batch = x_batch.to(self.model.device)
            y_batch = y_batch.to(self.model.device)

            pred = self.model.forward(x_batch)
            loss = criterion(pred, y_batch)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            # Clip the update from beginning of the step to now
            updated_params = self.model.flatten()
            self.model.unflatten(initial_params + flat_clip(updated_params - initial_params, self.S))

    def init(self, params):
        self.model.unflatten(params)

    def train(self, epochs):
        initial_params = self.model.flatten()
        for i in range(epochs):
            self._train_epoch(initial_params)

        updated_params = self.model.flatten()
        # Model is updated based on global model, i.e, it will be updated in the update method
        # This method just calculates the local update
        self.model.unflatten(initial_params)

        # Return already clipped user local update
        return updated_params - initial_params

    def update(self, update):
        # Apply the given update
        self.model.unflatten(self.model.flatten() + update)


class LaplaceMechanismStep:
    def __init__(self, S, epsilon):
        self.S = S
        self.epsilon = epsilon

        self.lambda_ = S / epsilon

    def _noise(self, shape):
        return np.random.laplace(loc=0.0, scale=self.lambda_, size=shape)

    def add_noise(self, update):
        return update + self._noise(update.shape)
