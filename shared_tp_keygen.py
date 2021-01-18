import argparse
import sys

from tp import ThresholdPaillier, write_key

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="")
    parser.add_argument('-s', '--size', default=1024, type=int)
    parser.add_argument('-c', '--clients', default=10, type=int)
    parser.add_argument('-t', '--threshold', default=5, type=int)
    args = parser.parse_args()

    key = ThresholdPaillier(args.size, args.clients, args.threshold)
    priv_keys = key.priv_keys
    pub_key = key.pub_key
    for i in range(args.clients):
        with open(f'tp_key_{i}.pkl', 'wb') as f:
            write_key(priv_keys[i], f)
    with open('tp_key_pub.pkl', 'wb') as f:
        write_key(pub_key, f)
