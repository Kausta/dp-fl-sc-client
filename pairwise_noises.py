from pyDH import DiffieHellman, primes

class PairwiseNoises:

    dh_instances: dict = None
    noise_map: dict = {}
    p = None

    def generate_private_keys(self, group_desc, clients):
        # Create Diffie Hellman instances for every other client.
        self.dh_instances = {client_id: DiffieHellman(group_desc) for client_id in clients.keys()}
        self.p = primes[group_desc]["prime"]

    def get_public_keys(self, clients):
        public_keys = {client_id: self.dh_instances[client_id].gen_public_key() for client_id in clients.keys()}
        return public_keys
    
    def receive_contribution(self, contribution, contributor_id):
        dh = self.dh_instances[contributor_id]
        self.noise_map[contributor_id] = dh.gen_shared_key(contribution)
    
    def update_noises(self):
        pass

    def get_noise(self, self_id):
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
        