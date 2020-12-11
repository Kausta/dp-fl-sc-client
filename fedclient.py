import torch
import torchvision
from fl_dp.models import MnistMLP
from fl_dp.strategies import LaplaceDpFed
from fl_dp.train import DpFedStep, LaplaceMechanismStep

from pairwise_noises import PairwiseNoises
import random


class FedClient:
    # Public parameters of the system. Received during initialization.
    public_params: dict = None
    noises = PairwiseNoises()
    initial_model = None
    client_id = None

    local_weight = 1000
    device = None
    dataset = None
    train_set = None
    test_set = None
    train_loader = None
    test_loader = None
    strategy = None

    # Invoked during registration.
    def set_id(self, client_id):
        self.client_id = client_id

    # Invoked during initialization (1).
    def set_public_parameters(self, public_params):
        print("FedClient: Received public parameters", public_params)
        self.public_params = public_params

    # Invoked during initialization (2).
    def load_data(self):
        print("FedClient: Loading data...")
        print("FedClient: Cuda:", torch.cuda.is_available())
        if torch.cuda.is_available():
            dev = "cuda:0"
        else:
            dev = "cpu"
        print("FedClient: Using device", dev)
        self.device = torch.device(dev)
        # Load all the data and choose only a part of them (for now.)
        self.train_set = torchvision.datasets.MNIST("data/mnist/", train=True, download=True,
                                               transform=torchvision.transforms.Compose([
                                                   torchvision.transforms.ToTensor(),
                                                   torchvision.transforms.Normalize(
                                                       (0.1307,), (0.3081,))
                                               ]))
        self.test_set = torchvision.datasets.MNIST("data/mnist/", train=False, download=True,
                                              transform=torchvision.transforms.Compose([
                                                  torchvision.transforms.ToTensor(),
                                                  torchvision.transforms.Normalize(
                                                      (0.1307,), (0.3081,))
                                              ]))
        # Choose 1000 random data points as my local data (for now.)
        indices = list(range(len(self.train_set)))
        my_indices = random.sample(indices, self.local_weight)
        self.dataset = torch.utils.data.Subset(self.train_set, my_indices)

    # Invoked during initialization (2).
    def load_learner(self):
        print("FedClient: Loading learner...")
        args = self.public_params
        # We assume each client has 1000 data points, for now!
        total_weight = self.local_weight * args["system_size"]
        batch_size_train = args['batch_size']
        batch_size_test = args['batch_size']
        self.train_loader = torch.utils.data.DataLoader(self.dataset, batch_size=batch_size_train, shuffle=True)
        self.test_loader = torch.utils.data.DataLoader(self.test_set, batch_size=batch_size_test, shuffle=True)
        model = MnistMLP(self.device)
        trainer = DpFedStep(model, self.train_loader, self.test_loader, args['lr'], args['S'])
        laplace_step = LaplaceMechanismStep(args['S'] / total_weight, args['epsilon'])
        self.strategy = LaplaceDpFed(trainer, laplace_step)

    # Invoked during initialization (3). Initial model should be flattened.
    def set_initial_model(self, initial_model):
        print("FedClient: Received initial model", initial_model)
        self.strategy.initialize(initial_model)
        print("FedClient: Initial model loaded.")

    # Invoked at the beginning of a round.
    def calculate_update(self):
        print("FedClient: Calculating update...")
        update = self.strategy.calculate_update(self.public_params['local_epochs'])
        print("FedClient: Updated.")
        return update, self.local_weight

    # Invoked at the end of a round.
    def apply_update(self, global_update):
        print("FedClient: Applying global update...")
        self.strategy.apply_update(global_update)
        print("FedClient: Applied.")
        self.strategy.test()

    # Returns the public keys for each client.
    def setup(self, clients):
        print("FedClient: Setting up...")
        self.noises.generate_private_keys(self.public_params["group_desc"], clients)
        # Convert the contributions into hex strings.
        r = {client_id: hex(contribution) for client_id, contribution in self.noises.get_public_keys(clients).items()}
        return r

    def receive_contribution(self, contributor_id: int, contribution: str):
        print("FedClient: Received a contribution from", contributor_id)
        # Save the contribution.
        self.noises.receive_contribution(contributor_id, int(contribution, 16))

    def get_noise_map(self):
        return self.noises.noise_map
