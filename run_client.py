import argparse
import os

import grpc
import numpy as np
import torch

import config
import tp
import helper
from fl_dp import key_util
from fl_dp.he import HEEncryptStep, HEDecryptStep
from fl_dp.models import MnistMLP
from fl_dp.paillier import PaillierEncryptStep, PaillierDecryptStep
from fl_dp.strategies import LaplaceDpFed, HELaplaceDpFed, PaillierLaplaceDpFed, MPCLaplaceDpFed, TPLaplaceDpFed
from fl_dp.train import DpFedStep, LaplaceMechanismStep
from mpc.mpc_step import MPCEncryptStep
from mpc.pairwise_noises import PairwiseNoises
from protocol import communication_pb2_grpc, communication_pb2 as pb2, util
from tp.tp_step import TPEncryptStep


def parse_client_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-s", "--server", type=str, help="Server address.", default="localhost:8000")
    parser.add_argument("-c", "--client-id", type=int, help="Client id.")
    args = parser.parse_args()
    return args.server, args.client_id


def create_dpfed(loader, test_loader, args, total_weight, device):
    model = MnistMLP(device)
    trainer = DpFedStep(model, loader, test_loader, args['lr'], args['S'])
    laplace_step = LaplaceMechanismStep(args['S'] / (len(loader)),
                                        (args['epsilon'] / args['global_epochs']))
    strategy = LaplaceDpFed(trainer, laplace_step)
    return strategy


def create_he(loader, test_loader, args, weight, total_weight, device, private_key, factor_exp):
    model = MnistMLP(device)
    trainer = DpFedStep(model, loader, test_loader, args['lr'], args['S'])
    laplace_step = LaplaceMechanismStep(args['S'] / (len(loader)),
                                        (args['epsilon'] / args['global_epochs']))
    he_encrypt = HEEncryptStep(private_key, factor_exp, weight)
    he_decrypt = HEDecryptStep(private_key, factor_exp, args['q'] * total_weight)
    strategy = HELaplaceDpFed(trainer, laplace_step, he_encrypt, he_decrypt)
    return strategy


def create_paillier(loader, test_loader, args, weight, total_weight, device, factor_exp, public_key, private_key):
    model = MnistMLP(device)
    trainer = DpFedStep(model, loader, test_loader, args['lr'], args['S'])
    laplace_step = LaplaceMechanismStep(args['S'] / (len(loader)),
                                        (args['epsilon'] / args['global_epochs']))
    paillier_encrypt = PaillierEncryptStep(public_key, factor_exp, weight)
    paillier_decrypt = PaillierDecryptStep(private_key, factor_exp, total_weight)
    strategy = PaillierLaplaceDpFed(trainer, laplace_step, paillier_encrypt, paillier_decrypt)
    return strategy


def create_smp(loader, test_loader, args, weight, total_weight, device, factor_exp, client_id, system_size, pn):
    model = MnistMLP(device)
    trainer = DpFedStep(model, loader, test_loader, args['lr'], args['S'])
    laplace_step = LaplaceMechanismStep(args['S'] / (len(loader) * system_size),
                                        (args['epsilon'] / args['global_epochs']))
    mpc_encrypt_step = MPCEncryptStep(factor_exp, weight, total_weight, client_id)
    mpc_encrypt_step.set_pairwise_noise_generator(pn)
    strategy = MPCLaplaceDpFed(trainer, laplace_step, mpc_encrypt_step)
    strategy.set_system_size(system_size)
    return strategy


def create_tp(loader, test_loader, args, weight, total_weight, device, factor_exp, tp_pub_key):
    model = MnistMLP(device)
    trainer = DpFedStep(model, loader, test_loader, args['lr'], args['S'])
    noise_factor = np.sqrt(tp_pub_key.l - tp_pub_key.w)
    laplace_step = LaplaceMechanismStep(args['S'] / (len(loader) * noise_factor),
                                        (args['epsilon'] / args['global_epochs']))
    mpc_encrypt_step = TPEncryptStep(factor_exp, weight, total_weight, tp_pub_key)
    strategy = TPLaplaceDpFed(trainer, laplace_step, mpc_encrypt_step)
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

    # This should work, especially when testing with multiple instances with gpus,
    # although my computer with a 1660S wasn't able to train 10 models at the same time
    # print("Cuda:", torch.cuda.is_available())
    # if torch.cuda.is_available():
    #    dev = "cuda:0"
    # else:
    #     dev = "cpu"
    dev = "cpu"
    print("Using device", dev)
    device = torch.device(dev)

    private_key, public_key = key_util.read_key(args['key_path'])

    if method == "dpfed":
        strategy = create_dpfed(train_loader, test_loader, args, total_weight, device)
    elif method == "he":
        strategy = create_he(train_loader, test_loader, args, weight, total_weight, device, private_key,
                             args['factor_exp'])
    elif method == "paillier":
        strategy = create_paillier(train_loader, test_loader, args, weight, total_weight, device, args['factor_exp'],
                                   public_key, private_key)

    elif method == "mpc":
        # Construct the pairwise noises
        # First, receive the system size.
        r = server_stub.GetSystemSize(pb2.VoidMsg())
        system_size = r.system_size
        print("Received system size", system_size)
        pn = PairwiseNoises()
        print("Generating private keys...")
        # Create a list of all the client ids.
        client_ids = list(range(system_size))
        client_ids.remove(client_id)
        pn.generate_private_keys(14, client_ids)
        public_keys = pn.get_public_keys(client_ids)
        print("Sending noise contributions from client", client_id)
        cont_iterator = map(lambda target_id: pb2.NoiseContribution(contributor_id=client_id,
                                                                    target_id=target_id,
                                                                    contribution=hex(public_keys[target_id])),
                            public_keys)
        received_contributions = server_stub.ForwardNoiseContributions(cont_iterator)
        print("Received noise contributions.")
        # Save the contributions.
        for cont in received_contributions:
            pn.receive_contribution(cont.contributor_id, int(cont.contribution, 16))
        # Initialize PRGs
        pn.initialize_prgs()
        strategy = create_smp(train_loader, test_loader, args, weight, total_weight, device, args['factor_exp'],
                              client_id, system_size, pn)
    elif method == "tp":
        tp_pub_key = tp.read_key('tp_key_pub.pkl')
        tp_priv_key = tp.read_key(f'tp_key_{client_id}.pkl')
        strategy = create_tp(train_loader, test_loader, args, weight, total_weight, device, args['factor_exp'],
                             tp_pub_key)
    else:
        return

    timer = helper.Timer('Total')
    data_sent_cnt = helper.Counter('Data Sent')
    data_received_cnt = helper.Counter('Data Received')

    strategy.initialize(init_model)
    acc = strategy.test()
    for comm_round in range(args['global_epochs']):
        print("Round", comm_round + 1)

        should_contribute = next(
            server_stub.ShouldContribute(pb2.ShouldContributeRequest(client_id=client_id, last_acc=acc)))
        if should_contribute.contribute:
            print("Chosen for training round. Training...")
            timer.start()
            update = strategy.calculate_update(args['local_epochs'])
            timer.stop()
            print("Trained. Committing the local encrypted update...")
            commit_update_request = pb2.CommitUpdateRequest(client_id=client_id, model=util.serialize_np(update))
            data_sent_cnt.add(commit_update_request.ByteSize())
            server_stub.CommitUpdate(commit_update_request)
            print("Committed.")
        else:
            print("Not chosen for round")

        if method == "tp":
            should_decrypt = next(
                server_stub.TpShouldPartialDecrypt(pb2.ShouldDecryptRequest(client_id=client_id)))
            data_received_cnt.add(should_decrypt.ByteSize())
            if should_decrypt.contribute:
                print("Choosen for partial decryption")
                timer.start()
                model = util.parse_np(should_decrypt.model)
                for i in range(len(model)):
                    model[i] = tp_priv_key.partialDecrypt(model[i])
                timer.stop()
                print("Decrypted. Committing the partial decryption...")
                commit_update_request = pb2.CommitUpdateRequest(client_id=client_id, model=util.serialize_np(model))
                data_sent_cnt.add(commit_update_request.ByteSize())
                server_stub.TpPartialDecrypt(commit_update_request)

        print("Waiting for the global model...")
        update = next(server_stub.GetGlobalUpdate(pb2.VoidMsg()))
        data_received_cnt.add(update.ByteSize())
        print("Received global model.")
        timer.start()
        strategy.apply_update(util.parse_np(update))
        timer.stop()
        print("Testing global model...")
        acc = strategy.test()

        print(timer)
        print(data_sent_cnt)
        print(data_received_cnt)


if __name__ == "__main__":
    serve_client()
