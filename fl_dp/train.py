import time

import numpy as np
import torch

from fl_dp.dpfed_model import DpFedModel
from fl_dp.helper import AverageMeter
from fl_dp.optimizer import SimpleSGDOptimizer


def projection(vec, S):
    return vec * np.min([1.0, S / np.linalg.norm(vec, ord=2)])


def flat_clip(vec, S):
    return projection(vec, S)


def per_layer_clip(layers, S):
    S_layer = S / np.sqrt(len(layers))
    return [projection(layer, S_layer) for layer in layers]


class DpFedStep:
    def __init__(self, model: DpFedModel, train_loader, test_loader, lr, S):
        self.model = model
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.optimizer = SimpleSGDOptimizer(self.model.parameters(), lr)

        self.lr = lr
        self.S = S

        self.epoch = 0

    def _train_epoch(self, initial_params, initial_layers):
        batch_time = AverageMeter()
        data_time = AverageMeter()
        losses = AverageMeter()

        criterion = torch.nn.NLLLoss()
        self.model.train()

        end = time.time()

        for i, (x_batch, y_batch) in enumerate(self.train_loader):
            data_time.update(time.time() - end)

            x_batch = x_batch.to(self.model.device)
            y_batch = y_batch.to(self.model.device)

            pred = self.model.forward(x_batch)
            loss = criterion(pred, y_batch)

            losses.update(loss.item(), y_batch.size(0))

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            # Clip the update from beginning of the step to now

            # # Flat
            # updated_params = self.model.flatten()
            # self.model.unflatten(initial_params + flat_clip(updated_params - initial_params, self.S))

            # # Per Layer
            updated_layers = self.model.get_layer_tensors()
            diff = [x1 - x2 for x1, x2 in zip(updated_layers, initial_layers)]
            clipped = per_layer_clip(diff, self.S)
            new = [x1 + x2 for x1, x2 in zip(initial_layers, clipped)]
            self.model.set_layer_tensors(new)

            batch_time.update(time.time() - end)
            end = time.time()
        print(
            'Epoch [{0}]:\t'
            'Time {batch_time.sum:.3f} ({batch_time.avg:.3f})\t'
            'Data {data_time.sum:.3f} ({data_time.avg:.3f})\t'
            'Loss {loss.val:.4f} ({loss.avg:.4f})\t'.format(self.epoch, batch_time=batch_time, data_time=data_time,
                                                            loss=losses),
            flush=True
        )

    def init(self, params):
        self.model.unflatten(params)

    def train(self, epochs):
        initial_params = self.model.flatten()
        initial_layers = self.model.get_layer_tensors(copy=True)
        for i in range(epochs):
            self.epoch += 1
            self._train_epoch(initial_params, initial_layers)

        updated_params = self.model.flatten()
        # Model is updated based on global model, i.e, it will be updated in the update method
        # This method just calculates the local update
        self.model.unflatten(initial_params)

        # Return already clipped user local update
        return updated_params - initial_params

    def update(self, update):
        # Apply the given update
        self.model.unflatten(self.model.flatten() + update)

    def test(self):
        batch_time = AverageMeter()
        losses = AverageMeter()

        criterion = torch.nn.NLLLoss()
        self.model.eval()

        total = 0
        correct = 0

        with torch.no_grad():
            end = time.time()
            for i, (x_batch, y_batch) in enumerate(self.test_loader):
                batch_size = len(y_batch)

                x_batch = x_batch.to(self.model.device)
                y_batch = y_batch.to(self.model.device)

                pred = self.model.forward(x_batch)
                loss = criterion(pred, y_batch)

                _, predicted = torch.max(pred.data, 1)
                total += y_batch.size(0)
                correct += (predicted == y_batch).sum().item()

                losses.update(loss.item(), batch_size)

                batch_time.update(time.time() - end)
                end = time.time()
            print(
                'Test:\t'
                'Time {batch_time.sum:.3f} ({batch_time.avg:.3f})\t'
                'Loss {loss.val:.4f} ({loss.avg:.4f})\t'.format(batch_time=batch_time,
                                                                loss=losses),
                flush=True
            )
            accuracy = correct / total
            print('Accuracy on the test dataset: {accuracy:.4f}%'.format(accuracy=100 * accuracy))
            return accuracy


class LaplaceMechanismStep:
    def __init__(self, sensitivity, epsilon):
        self.sensitivity = sensitivity
        self.epsilon = epsilon

        self.lambda_ = sensitivity / epsilon

    def _noise(self, shape):
        return np.random.laplace(loc=0.0, scale=self.lambda_, size=shape)

    def add_noise(self, update):
        return update + self._noise(update.shape)
