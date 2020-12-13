import argparse
import os

import torch
import torchvision
import numpy as np


def load_mnist(datadir):
    train_set = torchvision.datasets.MNIST(datadir, train=True, download=True,
                                           transform=torchvision.transforms.Compose([
                                               torchvision.transforms.ToTensor(),
                                               torchvision.transforms.Normalize(
                                                   (0.1307,), (0.3081,))
                                           ]))
    test_set = torchvision.datasets.MNIST(datadir, train=False, download=True,
                                          transform=torchvision.transforms.Compose([
                                              torchvision.transforms.ToTensor(),
                                              torchvision.transforms.Normalize(
                                                  (0.1307,), (0.3081,))
                                          ]))
    return train_set, test_set


def main():
    parser = argparse.ArgumentParser(description="")
    parser.add_argument('--data-dir', default="data/mnist/", type=str)
    parser.add_argument('-d', '--dataset', default="mnist", type=str)
    parser.add_argument('-c', '--clients', default=10, type=int)
    args = parser.parse_args()

    dataset = args.dataset.strip().lower()
    if dataset not in ('mnist',):
        parser.error(f"Incorrect dataset {dataset}")

    clients = args.clients
    if clients <= 0:
        parser.error(f"Incorrect client count {dataset}")

    if dataset == 'mnist':
        train_set, test_set = load_mnist(args.data_dir)
    else:
        return

    indices = list(range(len(train_set)))
    np.random.shuffle(indices)
    indices_sets = np.array_split(indices, clients)

    for i, index_set in enumerate(indices_sets):
        dataset = torch.utils.data.Subset(train_set, index_set)
        with open(os.path.join(args.data_dir, f'client-{i}.pt'), 'wb') as f:
            torch.save((dataset, test_set), f)


if __name__ == '__main__':
    main()
