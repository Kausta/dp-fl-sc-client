import numpy as np
import phe

# Encryption/Decryption from "Paillier Partially Homomorphic Encryption"
from fl_dp.key_util import PublicKey, PrivateKey


class PaillierEncryptStep:
    def __init__(self, pk: PublicKey):
        self.N = pk.N

        self.public_key = phe.PaillierPublicKey(self.N)

    def encrypt(self, update):
        return np.array([self.public_key.encrypt(x) for x in update])


class PaillierDecryptStep:
    def __init__(self, pk: PrivateKey):
        self.p = pk.p
        self.q = pk.q
        self.N = pk.N

        self.public_key = phe.PaillierPublicKey(self.N)
        self.private_key = phe.PaillierPrivateKey(self.public_key, self.p, self.q)

    def decrypt(self, update):
        return np.array([self.private_key.decrypt(x) for x in update])
