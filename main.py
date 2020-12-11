import torch
import torchvision
import numpy as np

import key_util
from he import HEEncryptStep, HEDecryptStep
from models import MnistMLP
from strategies import LaplaceDpFed, HELaplaceDpFed
from train import DpFedStep, LaplaceMechanismStep


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

    batch_size_train = args['batch_size']
    batch_size_test = args['batch_size']

    train_set = torchvision.datasets.MNIST(args['data_dir'], train=True, download=True,
                                           transform=torchvision.transforms.Compose([
                                               torchvision.transforms.ToTensor(),
                                               torchvision.transforms.Normalize(
                                                   (0.1307,), (0.3081,))
                                           ]))
    test_set = torchvision.datasets.MNIST(args['data_dir'], train=False, download=True,
                                          transform=torchvision.transforms.Compose([
                                              torchvision.transforms.ToTensor(),
                                              torchvision.transforms.Normalize(
                                                  (0.1307,), (0.3081,))
                                          ]))

    N_CLIENTS = 10

    indices = list(range(len(train_set)))
    np.random.shuffle(indices)
    indices_sets = np.array_split(indices, N_CLIENTS)
    weights = [len(x) for x in indices_sets]
    max_weight = np.max(weights)
    weights = [w / max_weight for w in weights]
    total_weight = np.sum(weights)
    print(total_weight, weights)

    train_loaders = []
    for set in indices_sets:
        dataset = torch.utils.data.Subset(train_set, set)
        train_loaders.append(torch.utils.data.DataLoader(dataset, batch_size=batch_size_train, shuffle=True))

    test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size_test, shuffle=True)

    private_key, public_key = key_util.read_key(args['key_path'])

    strategies = []
    for weight, loader in zip(weights, train_loaders):
        strategies.append(create_he(loader, test_loader, args, weight, total_weight, device, private_key, args['factor_exp']))

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
