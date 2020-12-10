from pyDH import DiffieHellman, primes

class PairwiseNoises:

    dh_instances: dict = None
    noise_map: dict = {}
    # Store the contributions that were received before the generation of private keys.
    unassigned_contributions: dict = {}
    p = None

    def generate_private_keys(self, group_desc: int, clients: dict):
        # Create Diffie Hellman instances for every other client.
        self.dh_instances = {client_id: DiffieHellman(group_desc) for client_id in clients.keys()}
        # Set the prime.
        self.p = primes[group_desc]["prime"]
        # Now, consume the unassigned contributions.
        for client_id, contribution in self.unassigned_contributions.items():
            self.receive_contribution(client_id, contribution)
        # Clear the unassigned contributions.
        self.unassigned_contributions = {}

    def get_public_keys(self, clients: dict):
        public_keys = {client_id: self.dh_instances[client_id].gen_public_key() for client_id in clients.keys()}
        return public_keys
    
    def receive_contribution(self, contributor_id: int, contribution: int):
        if contributor_id in self.noise_map:
            print("PairwiseNoises: Contribution from " + str(contributor_id) + " already received!")
        # If the private keys are not yet generated, save the contribution.
        if (self.dh_instances is None) or (contributor_id not in self.dh_instances):
            self.unassigned_contributions[contributor_id] = contribution
        else:
            # If the private keys were generated, calculate the pairwise noise.
            dh = self.dh_instances[contributor_id]
            self.noise_map[contributor_id] = dh.gen_shared_key(contribution)
    
    def update_noises(self):
        pass

    def get_noise(self, self_id: int):
        smaller_ids = filter(lambda k, v: k < self_id, self.noise_map.items())
        larger_ids = filter(lambda k, v: k > self_id, self.noise_map.items())

        neg_noises = map(lambda k, v: v, smaller_ids)
        pos_noises = map(lambda k, v: v, larger_ids)

        acc = 0
        for n in pos_noises:
            acc = (acc + n) % self.p
        for n in neg_noises:
            acc = (acc - n) % self.p

        return acc