# Comparative Analysis of Federated Learning Systems with Encrypted Computation

- By Caner Korkmaz (ckorkmaz16@ku.edu.tr), Ali Utkan Sahin (asahin17@ku.edu.tr)

## Instructions

Currently, DPFED (Differentially Private Federated Averaging) and 
HE (Differentially Private Federated Averaging with Homomorphic Encryption) are the algorithms that can be tested.
In the commands below, change the method to either `dpfed` or `he` test a specific one.
The conda environment required to run the code is provided in the file [environment.yml](./environment.yml).

First, data needs to be generated for each algorithm.
Assuming `C` is the number of clients that will participate (we initially tested with C=10)
```bash
python generate_client_data.py --clients C
```
Then, a server and clients can be run on the same instance as following
(there is a server host parameter for run_client.py that also allows seperate instances, not included here for simplicity)
```bash
# Each of the following commands are executed separately

# Run the server (--method he for HE algorithm)
python run_server.py --method dpfed

# Run each of the clients (Assuming C=10)
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

Afterwards, when all clients are saying `Waiting for initialization`, `s` can be pressed on the server to show which clients joined
and `r` can be pressed to start the algorithm. After it achieves a specified target accuracy on test dataset and the server prints `Target accuracy reached`, the clients automatically close,
and the server can be closed by pressing `q`. Accuracy obtained after each round can be observed in the server output.

The configuration for the test (not including `C` number of clients from above) is currently contained in [config.py](./config.py), 
and can be changed there, but it will be passed as program arguments in future. Different keys for HE other than the provided test key
can be generated using [keygen.py](./keygen.py)