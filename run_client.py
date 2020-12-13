import os
from time import time

import torch
import config
import grpc
import argparse

from fl_dp import key_util
from fl_dp.he import HEEncryptStep, HEDecryptStep
from fl_dp.models import MnistMLP
from fl_dp.strategies import LaplaceDpFed, HELaplaceDpFed
from fl_dp.train import DpFedStep, LaplaceMechanismStep
from protocol import communication_pb2_grpc, communication_pb2 as pb2, util


def parse_client_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-s", "--server", type=str, help="Server address.", default="localhost:8000")
    parser.add_argument("-c", "--client-id", type=int, help="Client id.")
    args = parser.parse_args()
    return args.server, args.client_id


def create_dpfed(loader, test_loader, args, total_weight, device):
    model = MnistMLP(device)
    trainer = DpFedStep(model, loader, test_loader, args['lr'], args['S'])
    laplace_step = LaplaceMechanismStep(args['S'] / (args['q'] * total_weight), args['epsilon'])
    strategy = LaplaceDpFed(trainer, laplace_step)
    return strategy


def create_he(loader, test_loader, args, weight, total_weight, device, private_key, factor_exp):
    model = MnistMLP(device)
    trainer = DpFedStep(model, loader, test_loader, args['lr'], args['S'])
    laplace_step = LaplaceMechanismStep(args['S'] / (args['q'] * total_weight), args['epsilon'])
    he_encrypt = HEEncryptStep(private_key, factor_exp, weight)
    he_decrypt = HEDecryptStep(private_key, factor_exp, args['q'] * total_weight)
    strategy = HELaplaceDpFed(trainer, laplace_step, he_encrypt, he_decrypt)
    return strategy


def serve_client():
    # Receive client arguments.
    server_addr, client_id = parse_client_args()
    channel = grpc.insecure_channel(server_addr, options=config.grpc_options())
    server_stub = communication_pb2_grpc.ServerStub(channel)

    args = config.get_config()

    with open(os.path.join(args['data_dir'], f'client-{client_id}.pt'), 'rb') as f:
        train_set, test_set = torch.load(f)
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=args['batch_size'], shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=args['batch_size'], shuffle=True)

    response = server_stub.RegisterClient(pb2.RegisterRequest(client_id=client_id, client_data_len=len(train_set)))
    print("Waiting for initialization")
    response = next(response)
    weight, total_weight, init_model, method = response.weight, response.total_weight, util.parse_np(
        response.model), response.method
    print("Using method", method)
    print(init_model)
    print(weight, total_weight)

    # This should work, although my computer with a 1660S wasn't able to train 10 models at the same time
    # print("Cuda:", torch.cuda.is_available())
    # if torch.cuda.is_available():
    #    dev = "cuda:0"
    # else:
    #     dev = "cpu"
    dev = "cpu"
    print("Using device", dev)
    device = torch.device(dev)
    if method == "dpfed":
        strategy = create_dpfed(train_loader, test_loader, args, total_weight, device)
    elif method == "he":
        private_key, public_key = key_util.read_key(args['key_path'])
        strategy = create_he(train_loader, test_loader, args, weight, total_weight, device, private_key,
                             args['factor_exp'])
    else:
        return

    strategy.initialize(init_model)
    acc = strategy.test()
    comm_round = 0
    while True:
        comm_round += 1
        print("Round", comm_round)

        should_contribute = next(
            server_stub.ShouldContribute(pb2.ShouldContributeRequest(client_id=client_id, last_acc=acc)))
        if should_contribute.finished:
            break
        if should_contribute.contribute:
            print("Chosen for training round")
            update = strategy.calculate_update(args['local_epochs'])
            server_stub.CommitUpdate(pb2.CommitUpdateRequest(client_id=client_id, model=util.serialize_np(update)))
        else:
            print("Not chosen for round")

        update = next(server_stub.GetGlobalUpdate(pb2.VoidMsg()))
        strategy.apply_update(util.parse_np(update))
        acc = strategy.test()

    print('Target accuracy', args['target_acc'], 'achieved at round', comm_round)


if __name__ == "__main__":
    serve_client()
