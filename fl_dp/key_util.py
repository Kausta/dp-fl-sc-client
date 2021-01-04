from Crypto.Math.Numbers import Integer
from Crypto.PublicKey import RSA


class PrivateKey:
    def __init__(self, p, q, N, p_inv, q_inv):
        self.p = p
        self.q = q
        self.N = N
        self.p_inv = p_inv
        self.q_inv = q_inv


class PublicKey:
    def __init__(self, N):
        self.N = N


def read_key(path):
    with open(path, 'r') as f:
        key = RSA.import_key(f.read())
    p, q, N = key.p, key.q, key.n
    p_inv = Integer(p).inverse(q)
    q_inv = Integer(q).inverse(p)
    return PrivateKey(int(p), int(q), int(N), int(p_inv), int(q_inv)), PublicKey(int(N))
