import numpy as np

# Encryption from "Towards Efficient and Privacy-preserving Federated Deep Learning"
from fl_dp.key_util import PrivateKey


class HEEncryptStep:
    def __init__(self, pk: PrivateKey, factor_exp, weight):
        self.p = pk.p
        self.q = pk.q
        self.N = pk.N
        self.p_inv = pk.p_inv
        self.q_inv = pk.q_inv

        self.p_p_inv = (self.p * self.p_inv) % self.N
        self.q_q_inv = (self.q * self.q_inv) % self.N

        self.weight = weight
        self.factor_exp = factor_exp

    def encrypt(self, update):
        update = (update * (10 ** self.factor_exp)).astype(np.int64)

        g_mu_p = update % self.p
        g_mu_q = update % self.q
        # print((self.q_q_inv * modular_pow(g_mu_p, self.p, self.N)) % self.N - (self.q_q_inv * g_mu_p) % self.N)
        # print((self.p_p_inv * modular_pow(g_mu_q, self.q, self.N)) % self.N)
        # print(((self.q_q_inv * modular_pow(g_mu_p, self.p, self.N)) % self.N + (
        #            self.p_p_inv * modular_pow(g_mu_q, self.q, self.N)) % self.N) % self.N)
        # Bottom two equations are mathematically equivalent, also results are always equivalent with both
        # However, the second one is computationally much less expensive
        # (Equivalence follows from p^-1 p = 1 mod q, q^-1 q = 1 mod p,
        #   g_mu_p ^ p = g_mu_p mod p, g_mu_q ^q = g_mu_q mod q,
        #   getting C = g_mu_p mod p, C = g_mu_q mod q, then applying CRT to get
        #   C = q^-1 q g_mu_p + p^-1 p g_mu_q mod (N=pq) )
        # They also provide equal results in practice, which can be tried with smaller vectors
        # c_mu = ((self.q_q_inv * modular_pow(g_mu_p, self.p, self.N)) % self.N +
        #         (self.p_p_inv * modular_pow(g_mu_q, self.q, self.N)) % self.N) % self.N
        c_mu = ((self.q_q_inv * g_mu_p) % self.N +
                (self.p_p_inv * g_mu_q) % self.N) % self.N

        return c_mu


# Decryption from "Towards Efficient and Privacy-preserving Federated Deep Learning"
class HEDecryptStep:
    def __init__(self, pk: PrivateKey, factor_exp, total_weight):
        self.p = pk.p
        self.q = pk.q
        self.N = pk.N
        self.m_p = pk.q_inv
        self.m_q = pk.p_inv

        self.m_p_q = (self.m_p * self.q) % self.N
        self.m_q_p = (self.m_q * self.p) % self.N

        self.total_weight = total_weight
        self.factor_exp = factor_exp

    def decrypt(self, update):
        # print(update)
        g_add_p = update % self.p
        g_add_q = update % self.q

        g_add = ((self.m_p_q * g_add_p) % self.N +
                 (self.m_q_p * g_add_q) % self.N) % self.N

        g_add[g_add > self.N // 2] -= self.N  # Convert negatives back
        g_add = (g_add.astype(np.float64) / (10 ** self.factor_exp))
        g_add = g_add / self.total_weight
        return g_add


def modular_pow(base, exponent, modulus):
    if modulus == 1:
        return np.zeros_like(base)
    result = np.ones_like(base, dtype=object)
    base = base % modulus
    while exponent > 0:
        if exponent % 2 == 1:
            result = (result * base) % modulus
        exponent = exponent // 2
        base = (base * base) % modulus
    return result
