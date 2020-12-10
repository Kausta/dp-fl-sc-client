from pairwise_noises import PairwiseNoises

class FedClient:
    # Public parameters of the system. Received during initialization.
    public_params: dict = None
    noises = PairwiseNoises()

    def set_public_parameters(self, public_params):
        self.public_params = public_params

    def setup(self, clients):
        self.noises.generate_private_keys(self.public_params["group_desc"], clients)
        

    def update(self, global_weight):
        pass