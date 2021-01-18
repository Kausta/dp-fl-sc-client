import numpy as np

from fl_dp import key_util
from fl_dp import paillier


def main():
    private, public = key_util.read_key('test_key.pem')
    enc = paillier.PaillierEncryptStep(public, 8, 1)
    dec = paillier.PaillierDecryptStep(private, 8, 1)

    def test_singular(x):
        x = np.array(x, dtype=np.float64)
        print("Val:", x)
        enc_x = enc.encrypt(x)
        print("Enc:", enc_x[0].c)
        dec_x = dec.decrypt(enc_x)
        print("Dec:", dec_x)
        print("========")

    test_singular([0] * 1000)
    test_singular([1])
    test_singular([-1])
    test_singular([0.5])
    test_singular([-0.5])
    test_singular([0.98])

    a = np.array([1], dtype=np.float64)
    b = np.array([-2], dtype=np.float64)
    c = np.array([-0.16], dtype=np.float64)
    print(a + b + c)
    print(dec.decrypt(enc.encrypt(a) + enc.encrypt(b) + enc.encrypt(c)))


if __name__ == '__main__':
    main()
