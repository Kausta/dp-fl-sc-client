import math
import pickle
import random

import sympy
from phe import paillier
from phe.util import powmod, invert


# Based on the Original Paillier paper Threshold Paillier paper,
# python phe paillier library and the website https://coderzcolumn.com/tutorials/python/threshold-paillier


class ThresholdPaillier(object):
    def __init__(self, size_of_n, clients, threshold):
        # size_of_n = 1024
        pub, priv = paillier.generate_paillier_keypair(n_length=size_of_n)
        self.p1 = priv.p
        self.q1 = priv.q

        while not sympy.ntheory.primetest.isprime(2 * self.p1 + 1):
            pub, priv = paillier.generate_paillier_keypair(n_length=size_of_n)
            self.p1 = priv.p
        while (not sympy.ntheory.primetest.isprime(2 * self.q1 + 1)) or self.p1 == self.q1:
            pub, priv = paillier.generate_paillier_keypair(n_length=size_of_n)
            self.q1 = priv.q

        self.p = (2 * self.p1) + 1
        self.q = (2 * self.q1) + 1
        self.n = self.p * self.q
        self.nSquare = self.n * self.n

        self.m = self.p1 * self.q1
        self.nm = self.n * self.m
        self.l = clients  # Number of shares of private key
        self.w = threshold  # The minimum of decryption servers needed to make a correct decryption.
        self.delta = math.factorial(self.l)
        self.combineSharesConstant = sympy.mod_inverse((4 * self.delta * self.delta) % self.n, self.n)
        self.d = self.m * sympy.mod_inverse(self.m, self.n)

        self.ais = [self.d]
        for i in range(1, self.w):
            self.ais.append(random.randint(0, self.nm - 1))

        self.si = [0] * self.l

        for i in range(self.l):
            self.si[i] = 0
            X = i + 1
            for j in range(self.w):
                self.si[i] += self.ais[j] * pow(X, j)
            self.si[i] = self.si[i] % self.nm

        self.priv_keys = []
        for i in range(self.l):
            self.priv_keys.append(ThresholdPaillierPrivateKey(self.n, self.l, self.combineSharesConstant, self.w,
                                                              self.si[i], i + 1,
                                                              self.delta, self.nSquare))
        self.pub_key = ThresholdPaillierPublicKey(self.n, self.nSquare, self.l, self.w,
                                                  self.delta, self.combineSharesConstant)


class PartialShare(object):
    def __init__(self, share, server_id):
        self.share = share
        self.server_id = server_id


class PaillierPublicKey(object):
    def __init__(self, n):
        self.g = n + 1
        self.n = n
        self.nsquare = n * n
        self.max_int = n // 3 - 1

    def get_random_lt_n(self):
        """Return a cryptographically random number less than :attr:`n`"""
        return random.SystemRandom().randrange(1, self.n)

    def get_random_r_to_n(self):
        return powmod(self.get_random_lt_n(), self.n, self.nsquare)

    def encrypt(self, plaintext, r_to_n_value=None):
        if self.n - self.max_int <= plaintext < self.n:
            # Very large plaintext, take a sneaky shortcut using inverses
            neg_plaintext = self.n - plaintext  # = abs(plaintext - nsquare)
            neg_ciphertext = (self.n * neg_plaintext + 1) % self.nsquare
            nude_ciphertext = invert(neg_ciphertext, self.nsquare)
        else:
            # we chose g = n + 1, so that we can exploit the fact that
            # (n+1)^plaintext = n*plaintext + 1 mod n^2
            nude_ciphertext = (self.n * plaintext + 1) % self.nsquare

        obfuscator = r_to_n_value or self.get_random_r_to_n()

        c = (nude_ciphertext * obfuscator) % self.nsquare
        return EncryptedNumber(c, self.nsquare, self.n)


class ThresholdPaillierPublicKey(object):
    def __init__(self, n, n_square, l, w, delta, combine_shares_constant):
        self.n = n
        self.nSquare = n_square
        self.l = l
        self.w = w
        self.delta = delta
        self.combineSharesConstant = combine_shares_constant

        self.pub = PaillierPublicKey(n)

        self.max_int = n // 3 - 1

    def get_random_lt_n(self):
        return self.pub.get_random_lt_n()

    def get_random_r_to_n(self):
        return self.pub.get_random_r_to_n()

    def encrypt(self, plaintext, r_to_n_value=None):
        return self.pub.encrypt(plaintext, r_to_n_value)

    def combinePartials(self, shrs):
        result = combineShares(shrs, self.w, self.delta, self.combineSharesConstant, self.nSquare, self.n)
        return result - self.n if result >= (self.n - self.max_int) else result


class PaillierPrivateKey(object):
    def __init__(self, n, p, q):
        if not p * q == n:
            raise ValueError('given public key does not match the given p and q.')
        if p == q:
            # check that p and q are different, otherwise we can't compute p^-1 mod q
            raise ValueError('p and q have to be different')

        self.g = n + 1
        self.n = n
        self.nsquare = n * n
        self.max_int = n // 3 - 1

        if q < p:  # ensure that p < q.
            self.p = q
            self.q = p
        else:
            self.p = p
            self.q = q
        self.psquare = self.p * self.p

        self.qsquare = self.q * self.q
        self.p_inverse = invert(self.p, self.q)
        self.hp = self.h_function(self.p, self.psquare)
        self.hq = self.h_function(self.q, self.qsquare)

    def decrypt(self, c):
        decrypt_to_p = self.l_function(powmod(c.c, self.p - 1, self.psquare), self.p) * self.hp % self.p
        decrypt_to_q = self.l_function(powmod(c.c, self.q - 1, self.qsquare), self.q) * self.hq % self.q
        result = self.crt(decrypt_to_p, decrypt_to_q)
        return result - self.n if result >= (self.n - self.max_int) else result

    def h_function(self, x, xsquare):
        return invert(self.l_function(powmod(self.g, x - 1, xsquare), x), x)

    def l_function(self, x, p):
        return (x - 1) // p

    def crt(self, mp, mq):
        u = (mq - mp) * self.p_inverse % self.q
        return mp + (u * self.p)


class ThresholdPaillierPrivateKey(object):
    def __init__(self, n, l, combine_shares_constant, w, si, server_id, delta, n_square):
        self.n = n
        self.l = l
        self.combineSharesConstant = combine_shares_constant
        self.w = w
        self.si = si
        self.server_id = server_id
        self.delta = delta
        self.nSquare = n_square

    def partialDecrypt(self, c):
        return PartialShare(powmod(c.c, self.si * 2 * self.delta, self.nSquare), self.server_id)


def write_key(key, file):
    pickle.dump(key, file)


def read_key(file):
    with open(file, 'rb') as file:
        return pickle.load(file)


class EncryptedNumber(object):
    def __init__(self, c, nSquare, n):
        self.c = c
        self.nSquare = nSquare
        self.n = n

    def __add__(self, c2):
        return EncryptedNumber((self.c * c2.c) % self.nSquare, self.nSquare, self.n)

    def __radd__(self, other):
        """Called when Python evaluates `34 + <EncryptedNumber>`
        Required for builtin `sum` to work.
        """
        return self.__add__(other)


def combineShares(shrs, w, delta, combineSharesConstant, nSquare, n):
    cprime = 1
    for i in range(w):
        ld = delta
        for iprime in range(w):
            if i != iprime:
                if shrs[i].server_id != shrs[iprime].server_id:
                    ld = (ld * -shrs[iprime].server_id) // (shrs[i].server_id - shrs[iprime].server_id)
        shr = sympy.mod_inverse(shrs[i].share, nSquare) if ld < 0 else shrs[i].share
        ld = -1 * ld if ld < 1 else ld
        temp = powmod(shr, 2 * ld, nSquare)
        cprime = (cprime * temp) % nSquare
    L = (cprime - 1) // n
    result = (L * combineSharesConstant) % n
    return result
