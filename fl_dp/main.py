import torch
import torchvision
import numpy as np

from models import MnistMLP
from strategies import LaplaceDpFed
from train import DpFedStep, LaplaceMechanismStep


def main():
    print("Cuda:", torch.cuda.is_available())
    if torch.cuda.is_available():
        dev = "cuda:0"
    else:
        dev = "cpu"
    print("Using device", dev)
    device = torch.device(dev)

    args = {
        'lr': 0.01,
        'S': 1,
        'epsilon': 1,
        'batch_size': 100,
        'local_epochs': 5,
        'data_dir': "data/mnist/",
        'target_acc': 0.95
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
    ROUNDS = 1

    indices = list(range(len(train_set)))
    np.random.shuffle(indices)
    indices_sets = np.array_split(indices, N_CLIENTS)
    weights = [len(x) for x in indices_sets]
    min_weight = np.min(weights)
    weights = [w / min_weight for w in weights]
    total_weight = np.sum(weights)
    print(total_weight, weights)

    train_loaders = []
    for set in indices_sets:
        dataset = torch.utils.data.Subset(train_set, set)
        train_loaders.append(torch.utils.data.DataLoader(dataset, batch_size=batch_size_train, shuffle=True))

    test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size_test, shuffle=True)

    strategies = []
    for loader in train_loaders:
        model = MnistMLP(device)
        trainer = DpFedStep(model, loader, test_loader, args['lr'], args['S'])
        laplace_step = LaplaceMechanismStep(args['S'] / total_weight, args['epsilon'])
        strategy = LaplaceDpFed(trainer, laplace_step)
        strategies.append(strategy)

    init_model = MnistMLP(device)
    # Initialize the clients with the initial model.
    for strategy in strategies:
        strategy.initialize(init_model.flatten())
        print(init_model.flatten())
    for i in range(ROUNDS):
        updates = []
        for client, strategy in enumerate(strategies):
            updates.append(strategy.calculate_update(args['local_epochs']))
        update = np.zeros_like(updates[0])

        for weight, local_update in zip(weights, updates):
            update += weight * local_update
        update /= total_weight

        for client, strategy in enumerate(strategies):
            strategy.apply_update(update)
        # 1 is sufficient
        strategies[0].test()


if __name__ == '__main__':
    main()
