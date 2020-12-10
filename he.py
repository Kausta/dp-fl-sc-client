import numpy as np


# Encryption from "Towards Efficient and Privacy-preserving Federated Deep Learning"
class HEEncryptStep:
    def __init__(self, p, q, N, p_inv, q_inv):
        self.p = p
        self.q = q
        self.N = N
        self.p_inv = p_inv
        self.q_inv = q_inv

        self.p_p_inv = (p * p_inv) % N
        self.q_q_inv = (q * q_inv) % N

    def encrypt(self, update):
        g_mu_p = update % self.p
        g_mu_q = update % self.q

        c_mu = ((self.q_q_inv * modular_pow(g_mu_p, self.p, self.N)) % self.N +
                (self.p_p_inv * modular_pow(g_mu_q, self.q, self.N)) % self.N) % self.N

        return c_mu


# Decryption from "Towards Efficient and Privacy-preserving Federated Deep Learning"
class HEDecryptStep:
    def __init__(self, p, q, N, m_p, m_q):
        self.p = p
        self.q = q
        self.N = N
        self.m_p = m_p
        self.m_q = m_q

        self.m_p_q = (m_p * q) % N
        self.m_q_p = (m_q * p) % N

    def decrypt(self, update):
        g_add_p = update % self.p
        g_add_q = update % self.q

        g_add = ((self.m_p_q * g_add_p) % self.N +
                 (self.m_q_p * g_add_q) % self.N) % self.N

        return g_add


# Taken from Applied Cryptography via Wikipedia
# https://en.wikipedia.org/w/index.php?title=Modular_exponentiation&oldid=761445844#Right-to-left_binary_method
def modular_pow(base, exponent, modulus):
    if modulus == 1:
        return np.zeros_like(base)
    result = np.ones_like(base)
    base = base % modulus
    while exponent > 0:
        if exponent % 2 == 1:
            result = (result * base) % modulus
        exponent = exponent >> 1
        base = (base * base) % modulus
    return result
