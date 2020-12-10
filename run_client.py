from client import Client
from communication import communication_pb2_grpc, rpc_client
from concurrent import futures
import grpc
import argparse

def parse_client_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("port", type=int, help="Listening port for the client.")
    parser.add_argument("-s", type=str, help="Server address.", default="localhost:8000")
    parser.add_argument("-w", type=int, help="Number of workers.", default=10)
    return parser.parse_args()

def serve_client():
    # Receive client arguments.
    args = parse_client_args()
    client_port = args.port
    server_addr = args.s
    client = Client("localhost:" + str(client_port), server_addr)
    s = grpc.server(futures.ThreadPoolExecutor(max_workers = args.w))
    communication_pb2_grpc.add_ClientServicer_to_server(rpc_client.RpcClient(client), s)
    s.add_insecure_port("[::]:" + str(client_port))
    s.start()
    print("Client - Listening on", str(client_port))
    # Start the client interface.
    client.run_interface()

if __name__ == "__main__":
    serve_client()