import numpy as np
import tp
# Encryption/Decryption from "Paillier Partially Homomorphic Encryption"
from fl_dp.key_util import PublicKey, PrivateKey


class PaillierEncryptStep:
    def __init__(self, pk: PublicKey, factor_exp, weight):
        self.weight = weight
        self.factor_exp = factor_exp

        self.public_key = tp.PaillierPublicKey(pk.N)
        self.noise_gen = tp.NoiseGenerator(self.public_key)

    def encrypt(self, update):
        update = self.weight * update
        update = (update * (10 ** self.factor_exp)).astype(np.int64)
        return np.array([self.public_key.encrypt(x, self.noise_gen) for x in update])


class PaillierDecryptStep:
    def __init__(self, pk: PrivateKey, factor_exp, total_weight):
        self.total_weight = total_weight
        self.factor_exp = factor_exp

        self.private_key = tp.PaillierPrivateKey(pk.N, pk.p, pk.q)

    def decrypt(self, update):
        update = np.array([self.private_key.decrypt(c) for c in update])
        return (update.astype(np.float64) / (10 ** self.factor_exp)) / self.total_weight
