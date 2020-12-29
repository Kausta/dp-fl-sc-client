from mpc.pairwise_noises import PairwiseNoises
import numpy as np


class MPCEncryptStep:
    pn = None

    def __init__(self, factor_exp, weight, self_id):
        self.weight = weight
        self.factor_exp = factor_exp
        self.id = self_id

    def set_pairwise_noise_generator(self, pn: PairwiseNoises):
        self.pn = pn

    def encrypt(self, update):
        update = self.weight * update
        update = (update * (10 ** self.factor_exp)).astype(np.int64)
        update += self.pn.get_noise(self.id)
        return update
