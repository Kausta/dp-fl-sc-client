import numpy as np

from . import ThresholdPaillierPublicKey


class TPEncryptStep:
    def __init__(self, factor_exp, weight, total_weight, public_key: ThresholdPaillierPublicKey):
        self.weight = weight
        self.factor_exp = factor_exp
        self.total_weight = total_weight
        self.public_key = public_key

    def encrypt(self, update):
        update = self.weight * update
        update = (update * (10 ** self.factor_exp)).astype(np.int64)
        r_to_n_val = self.public_key.get_random_r_to_n()
        return np.array([self.public_key.encrypt(x, r_to_n_val) for x in update])

    def decrypt_global_update(self, update):
        return (update.astype(np.float64) / (10 ** self.factor_exp)) / self.total_weight
