import numpy as np

from mpc.pairwise_noises import PairwiseNoises


class MPCEncryptStep:
    pn = None

    def __init__(self, factor_exp, weight, total_weight, self_id):
        self.weight = weight
        self.factor_exp = factor_exp
        self.total_weight = total_weight
        self.id = self_id

    def set_pairwise_noise_generator(self, pn: PairwiseNoises):
        self.pn = pn

    def encrypt(self, update):
        update = self.weight * update
        update = (update * (10 ** self.factor_exp)).astype(np.int64)
        noise = self.pn.get_noise(self.id)
        res = update + noise
        return res

    def decrypt_global_update(self, update, system_size):
        return ((update.astype(np.float128) / (10 ** self.factor_exp)) / self.total_weight).astype(np.float64)
