import argparse
import os

import torch
import torchvision
import numpy as np

from fl_dp import key_util
from fl_dp.he import HEEncryptStep, HEDecryptStep
from fl_dp.models import MnistMLP
from fl_dp.strategies import LaplaceDpFed, HELaplaceDpFed
from fl_dp.train import DpFedStep, LaplaceMechanismStep


def choose_randomly(arr, p):
    for i, elem in enumerate(arr):
        if np.random.random_sample() <= p:
            yield i, elem


def create_dpfed(loader, test_loader, args, total_weight, device):
    model = MnistMLP(device)
    trainer = DpFedStep(model, loader, test_loader, args['lr'], args['S'])
    laplace_step = LaplaceMechanismStep(args['S'] / (args['q'] * total_weight), args['epsilon'])
    strategy = LaplaceDpFed(trainer, laplace_step)
    return strategy


def average_dpfed(updates, args, weights, total_weight):
    update = np.zeros_like(updates[0])
    for weight, local_update in zip(weights, updates):
        update += weight * local_update
    update /= (args['q'] * total_weight)
    return update


def create_he(loader, test_loader, args, weight, total_weight, device, private_key, factor_exp):
    model = MnistMLP(device)
    trainer = DpFedStep(model, loader, test_loader, args['lr'], args['S'])
    laplace_step = LaplaceMechanismStep(args['S'] / (args['q'] * total_weight), args['epsilon'])
    he_encrypt = HEEncryptStep(private_key, factor_exp, weight)
    he_decrypt = HEDecryptStep(private_key, factor_exp, args['q'] * total_weight)
    strategy = HELaplaceDpFed(trainer, laplace_step, he_encrypt, he_decrypt)
    return strategy


def average_he(updates, public_key):
    update = np.zeros_like(updates[0])
    for local_update in updates:
        update = (update + local_update)
    return update


def main():
    parser = argparse.ArgumentParser(description="")
    parser.add_argument('-m', '--method', default="dpfed", type=str)
    args = parser.parse_args()
    method = args.method.strip().lower()
    if method not in ('dpfed', 'he'):
        parser.error(f"Incorrect strategy {method}")

    print("Cuda:", torch.cuda.is_available())
    if torch.cuda.is_available():
        dev = "cuda:0"
    else:
        dev = "cpu"
    print("Using device", dev)
    device = torch.device(dev)

    args = {
        'data_dir': "data/mnist/",
        'key_path': 'test_key.pem',
        'factor_exp': 16,
        'lr': 0.01,
        'S': 1,
        'epsilon': 1,
        'batch_size': 100,
        'local_epochs': 5,
        'target_acc': 0.90,
        'q': 1
    }

    N_CLIENTS = 10

    datasets = []
    test_set = None
    for i in range(N_CLIENTS):
        with open(os.path.join(args['data_dir'], f'client-{i}.pt'), 'rb') as f:
            dataset, test_set_ = torch.load(f)
            datasets.append(dataset)
            if test_set is None:
                test_set = test_set_

    weights = [len(x) for x in datasets]
    max_weight = np.max(weights)
    weights = [w / max_weight for w in weights]
    total_weight = np.sum(weights)
    print(total_weight, weights)

    train_loaders = []
    for dataset in datasets:
        print(len(dataset))
        train_loaders.append(torch.utils.data.DataLoader(dataset, batch_size=args['batch_size'], shuffle=True))

    test_loader = torch.utils.data.DataLoader(test_set, batch_size=args['batch_size'], shuffle=True)

    private_key, public_key = key_util.read_key(args['key_path'])

    strategies = []
    for weight, loader in zip(weights, train_loaders):
        if method == "dpfed":
            strategies.append(
                create_dpfed(loader, test_loader, args, total_weight, device))
        elif method == "he":
            strategies.append(
                create_he(loader, test_loader, args, weight, total_weight, device, private_key, args['factor_exp']))

    init_model = MnistMLP(device)
    for strategy in strategies:
        strategy.initialize(init_model.flatten())
    comm_round = 0
    while True:
        comm_round += 1
        print("Round", comm_round)

        chosen = list(choose_randomly(strategies, args['q']))
        print(f"Chosen clients for round: [{', '.join([str(x) for x, _ in chosen])}]")

        updates = []
        for client, strategy in chosen:
            print("Training client", client)
            updates.append(strategy.calculate_update(args['local_epochs']))

        if method == "dpfed":
            update = average_dpfed(updates, args, weights, total_weight)
        elif method == "he":
            update = average_he(updates, public_key)

        for client, strategy in enumerate(strategies):
            strategy.apply_update(update)
        # 1 is sufficient
        acc = strategies[0].test()
        if acc >= args['target_acc']:
            print('Target accuracy', args['target_acc'], 'achieved at round', comm_round)
            break


if __name__ == '__main__':
    main()
