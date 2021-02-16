# Comparative Analysis of Federated Learning Systems with Encrypted Computation

- By Caner Korkmaz (ckorkmaz16@ku.edu.tr), Ali Utkan Sahin (asahin17@ku.edu.tr)

## Introduction

The algorithms implemented are as following, with their method names in the application:
- Differentially Private Federated Averaging (`dpfed`)
- Paillier Homomorphic Encryption (`paillier`)
- Towards Efficient and Privacy-Preserving Federated Deep Learning, referred as Custom HE (`he`)
- Differentially Private Secure Multi-Party Computation for  Federated Learning in Financial Applications (`mpc`)
- A Hybrid Approach to Privacy-Preserving Federated Learning, referred as Threshold Paillier (`tp`)

The general methodology, gradient clipping strategy and noise strategy is the same for all of the algorithms to
compare them fairly, with noise scale and encryption/decryption steps changing.

## Setup

The project uses Python 3.8 with conda. A conda installation is required to install the libraries with the shared files.
In Windows, [environment.yml](./environment.yml) can be used with the following command to install and activate the environment:
```bash
conda env create -f environment.yml
conda activate comp430
```
The same commands may also be used in other OS's, but may not work. In that case, a Python 3.8 environment with the following libraries are required:
`grpcio`, `grpcio-tools`, `phe`, `numpy`, `scipy`, `sympy`, `scikit-learn`, `pytorch`, `torchvision`, `cudatoolkit=10.2` (except on MacOS), `gmpy2`,
`pandas`, `pycryptodome`.

Then, client data, private key for `paillier` and `he`, and secret key shares for `tp` needs to be generated. Assuming there are 10 clients, and the threshold for `tp` is 5, the commands required are as following:
```bash
python generate_client_data.py --clients 10
python keygen.py
python shared_tp_keygen.py --clients 10 --threshold 5
```

## Running

We need to run a server and clients. There is a host parameter for the clients (`-s`), but we tested on the same instance, 
and for simplicity, the below are for that case:

```bash
# Each of the following commands are executed in separate terminals

# Run the server (--method can be dpfed | paillier | he | mpc | tp )
python run_server.py --method dpfed

# Run each of the clients (Assuming there are 10 clients)
python run_client.py -c 0
python run_client.py -c 1
python run_client.py -c 2
python run_client.py -c 3
python run_client.py -c 4
python run_client.py -c 5
python run_client.py -c 6
python run_client.py -c 7
python run_client.py -c 8
python run_client.py -c 9
```

Afterwards, when all clients are saying `Waiting for initialization`, `s` can be pressed on the server to show which
clients joined and `r` can be pressed to start the algorithm. After the configured amount of rounds pass, clients close
automatically, and the server can be stopped by pressing `q` (different runs requires restarting the server).
Accuracy obtained after each round can be observed in the client outputs, and accuracies except for the last round can also be observed in the server output.
Moreover, clients and server report total execution times, and clients also report incoming and outgoing total uncompressed message sizes.
The reported times don't include initialization times.


The configuration for the test (not including `C` number of clients from above) is contained
in [config.py](./config.py), and can be changed there. These could have been arguments, but for simplicity, we put the arguments we used for our benchmark
in that file and used that in both server and client.

## License

Copyright © 2020 Caner Korkmaz, Utkan Şahin

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

   http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
