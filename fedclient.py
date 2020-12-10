from pairwise_noises import PairwiseNoises

class FedClient:
    # Public parameters of the system. Received during initialization.
    public_params: dict = None
    noises = PairwiseNoises()

    def set_public_parameters(self, public_params):
        self.public_params = public_params

    # Returns the public keys for each client.
    def setup(self, clients):
        self.noises.generate_private_keys(self.public_params["group_desc"], clients)
        # Convert the contributions into hex strings.
        r = {client_id: hex(contribution) for client_id, contribution in self.noises.get_public_keys(clients).items()}
        return r
    
    def receive_contribution(self, contributor_id: int, contribution: str):
        # Save the contribution.
        self.noises.receive_contribution(contributor_id, int(contribution, 16))
        print("FedClient: Received a contribution from", contributor_id)
    
    def get_noise_map(self):
        return self.noises.noise_map
    
    def update(self, global_weight):
        pass