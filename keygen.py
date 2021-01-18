import argparse
import sys

from Crypto.PublicKey import RSA

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="")
    parser.add_argument('-p', '--path', default="test_key.pem", type=str)
    parser.add_argument('-s', '--size', default=1024, type=int)
    args = parser.parse_args()

    key = RSA.generate(args.size)
    with open(args.path, 'wb') as f:
        f.write(key.export_key('PEM'))
