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
    return parser.parse_args()


def serve_server():
    args = config.get_config()
    initial_model = MnistMLP()
    server = rpc_server.Server(initial_model.flatten(), args)

    args = parse_server_args()
    server_port = args.port
    s = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    communication_pb2_grpc.add_ServerServicer_to_server(rpc_server.RpcServer(server), s)
    s.add_insecure_port("[::]:" + str(server_port))
    s.start()
    print("Server - Listening on", server_port)

    user_input = ""
    while user_input != "q":
        print(server_menu)
        user_input = input("Choose an option:")
        if user_input == "r":
            Thread(target=lambda: server.run()).start()
        elif user_input == "s":
            print(", ".join(map(str, server.client_list)))
    os._exit(0)


if __name__ == "__main__":
    serve_server()
