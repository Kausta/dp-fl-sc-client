import argparse
import grpc
from concurrent import futures

from communication import communication_pb2_grpc
from communication import rpc_server
from smpc_server import Server


def parse_server_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("port", type=int, help="Listening port for the server.")
    parser.add_argument("-w", type=int, help="Number of workers.", default=10)
    return parser.parse_args()


def serve_server():
    server = Server()
    args = parse_server_args()
    server_port = args.port
    s = grpc.server(futures.ThreadPoolExecutor(max_workers=args.w))
    communication_pb2_grpc.add_ServerServicer_to_server(rpc_server.RpcServer(server), s)
    s.add_insecure_port("[::]:" + str(server_port))
    s.start()
    print("Server - Listening on", server_port)
    server.run_interface()


if __name__ == "__main__":
    serve_server()
