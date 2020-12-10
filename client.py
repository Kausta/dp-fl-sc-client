from pyDH import DiffieHellman, primes
import random
from concurrent import futures

import grpc
from communication import communication_pb2
from communication import communication_pb2_grpc
from communication import rpc_client
import public_parameters

class Client:
    serverstub = None
    clientinfo = None
    clientmap = {}

    self_addr: str = None
    public_parameters: dict = None

    def __init__(self, self_addr):
        self.self_addr = self_addr

    def register(self, serveraddr):
        channel = grpc.insecure_channel(serveraddr)
        self.serverstub = communication_pb2_grpc.ServerStub(channel)
        print("Registering with the server at " + serveraddr + "...")
        self.clientinfo = self.serverstub.RegisterClient(communication_pb2.ClientAddress(client_address=self.self_addr))
        print("Registered with id", self.clientinfo.client_id)
        # Receive client list.
        for clientinfo in self.serverstub.ReceiveClients(communication_pb2.VoidMsg()):
            if clientinfo.client_id == self.clientinfo.client_id:
                continue
            self.clientmap[clientinfo.client_id] = clientinfo.client_address
        print("Received client list of size", len(self.clientmap))

    # Called when a new client is joined to the system.
    def new_client(self, new_client_id, new_client_addr):
        print("New client", new_client_addr, "joined with id", new_client_id)
        # Save the client.
        self.clientmap[new_client_id] = new_client_addr
    
    # Process the public parameters received from the server.
    def process_public_parameters(self, public_params):
        print("Processing the public parameters...")
        self.public_parameters = public_params
    
    # Invoked by the server at the beginning of the initialization phase.
    def initialize(self, public_parameters):
        print("Initializing the client...")
        print("Received public parameters:", public_parameters)
        self.process_public_parameters(public_parameters)
        # Initialize here.
        print("Initialized.")
    
    def client_update(self, global_model: list):
        # Minimize Err(global_model)
        return (global_model, self.clientinfo.client_id)

def serve_client():
    import sys
    client_port = sys.argv[1]
    server_addr = sys.argv[2]
    client = Client("localhost:" + client_port)
    s = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    communication_pb2_grpc.add_ClientServicer_to_server(rpc_client.RpcClient(client), s)
    s.add_insecure_port("[::]:" + client_port)
    # Register the client with the server.
    client.register(server_addr)
    s.start()
    s.wait_for_termination()

if __name__ == "__main__":
    serve_client()