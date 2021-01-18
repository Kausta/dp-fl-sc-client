import numpy as np

from . import ThresholdPaillierPublicKey, NoiseGenerator


class TPEncryptStep:
    def __init__(self, factor_exp, weight, total_weight, public_key: ThresholdPaillierPublicKey):
        self.weight = weight
        self.factor_exp = factor_exp
        self.total_weight = total_weight
        self.public_key = public_key

        self.noise_gen = NoiseGenerator(self.public_key.pub, 16)

    def encrypt(self, update):
        update = self.weight * update
        update = (update * (10 ** self.factor_exp)).astype(np.int64)
        return np.array([self.public_key.encrypt(x, self.noise_gen) for x in update])

    def decrypt_global_update(self, update):
        return (update.astype(np.float64) / (10 ** self.factor_exp)) / self.total_weight
