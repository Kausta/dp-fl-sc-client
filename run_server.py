import os
from threading import Thread

import config
from fl_dp.models import MnistMLP
from protocol import communication_pb2_grpc
from protocol import rpc_server

from concurrent import futures
import grpc

import argparse

server_menu = "\n(s)how the clients\n(r)un\n(q)uit"


def parse_server_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8000, help="Listening port for the server.")
    parser.add_argument('-m', '--method', default="dpfed", type=str)
    args = parser.parse_args()
    if args.method not in ('dpfed', 'he', 'mpc'):
        parser.error(f"Incorrect strategy {args.method}")
    return args.port, args.method


def serve_server():
    server_port, method = parse_server_args()
    print("Using method", method)

    args = config.get_config()
    initial_model = MnistMLP()
    server = rpc_server.Server(initial_model.flatten(), args, method)
    s = grpc.server(futures.ThreadPoolExecutor(max_workers=10), options=config.grpc_options())
    communication_pb2_grpc.add_ServerServicer_to_server(rpc_server.RpcServer(server), s)
    s.add_insecure_port("[::]:" + str(server_port))
    s.start()
    print("Server - Listening on", server_port)

    user_input = ""
    show_menu = True
    while user_input != "q":
        if show_menu:
            print(server_menu)
        user_input = input("Choose an option:")
        if user_input == "r":
            Thread(target=lambda: server.run()).start()
            show_menu = False
        elif user_input == "s":
            print(", ".join(map(str, server.client_list)))
    os._exit(0)


if __name__ == "__main__":
    serve_server()
