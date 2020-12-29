from mpc.pyDH import DiffieHellman, primes
import random


class PairwiseNoises:
    dh_instances: dict = None
    noise_map: dict = {}
    # Store the contributions that were received before the generation of private keys.
    unassigned_contributions: dict = {}
    # PRGs
    prgs: dict = {}
    p = None
    max_pairs: int = 0

    def generate_private_keys(self, group_desc: int, client_ids: list):
        self.max_pairs = len(client_ids)
        # Create Diffie Hellman instances for every other client.
        self.dh_instances = {client_id: DiffieHellman(group_desc) for client_id in client_ids}
        # Set the prime.
        self.p = primes[group_desc]["prime"]
        # Now, consume the unassigned contributions.
        for client_id, contribution in self.unassigned_contributions.items():
            self.receive_contribution(client_id, contribution)
        # Clear the unassigned contributions.
        self.unassigned_contributions = {}

    def get_public_keys(self, client_ids: list):
        public_keys = {client_id: self.dh_instances[client_id].gen_public_key() for client_id in client_ids}
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
        # If we have calculated all possible noises, initialize the PRGs.
        if len(self.noise_map) == self.max_pairs:
            self.initialize_prgs()

    def initialize_prgs(self):
        print("PairwiseNoises: Initializing PRGs...")
        self.prgs = {client_id: random.Random(shared_noise) for client_id, shared_noise in self.noise_map.items()}
        self.update_noises()

    # Updates the noises.
    def update_noises(self):
        print("PairwiseNoises: Updating noises...")
        for client_id in self.noise_map.keys():
            self.noise_map[client_id] = self.prgs[client_id].randint(0, self.p)

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
