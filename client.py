import random
from concurrent import futures

import grpc
import fedclient
from communication import communication_pb2
from communication import communication_pb2_grpc
import public_parameters

client_menu = "\n(s)how clients\nshow (p)airwise noises\n(q)uit"

class Client:
    # The server RPC connection.
    server_stub = None
    # The information of this client.
    client_info = None
    # Maps a client id to its address.
    clients = {}
    # Address of this client.
    self_addr: str = None
    # Address of the server.
    server_addr: str = None
    # The public parameters received by this client.
    public_parameters: dict = None

    # Federated client.
    fed_client = fedclient.FedClient()

    def __init__(self, self_addr, server_addr):
        self.self_addr = self_addr
        self.server_addr = server_addr
    
    def run_interface(self):
        # Register with the server preemptively.
        self.register()
        user_input = ""
        while user_input != "q":
            print(client_menu)
            user_input = input("Choose an option:")
            if user_input == "s":
                for client_id, client_addr in self.clients.items():
                    print(str(client_id) + ": " + client_addr)
            elif user_input == "p":
                for client_id, shared_noise in self.fed_client.get_noise_map().items():
                    print(str(client_id) + ": " + hex(shared_noise)[:10])

    # Registers with the server at the given address.
    def register(self):
        # Create the connection with the server.
        channel = grpc.insecure_channel(self.server_addr)
        self.server_stub = communication_pb2_grpc.ServerStub(channel)
        print("\nRegistering with the server at " + self.server_addr + "...")
        self.client_info = self.server_stub.RegisterClient(communication_pb2.ClientAddress(client_address=self.self_addr))
        print("Registered with id", self.client_info.client_id)
        # Receive client list.
        for client_info in self.server_stub.ReceiveClients(communication_pb2.VoidMsg()):
            if client_info.client_id == self.client_info.client_id:
                continue
            self.clients[client_info.client_id] = client_info.client_address
        print("Received client list of size", len(self.clients))

    # Called when a new client is joined to the system.
    def new_client(self, new_client_id: int, new_client_addr: str):
        print("\nNew client", new_client_addr, "joined with id", new_client_id)
        # Save the client.
        self.clients[new_client_id] = new_client_addr
    
    # Invoked by the server at the beginning of the setup phase.
    def setup(self):
        print("\nSet up phase initiated...")
        pub_keys = self.fed_client.setup(self.clients)
        print("Forwarding contributions through the server...")
        self.server_stub.ForwardContributions(public_parameters.serialize_contributions(pub_keys, self.client_info.client_id))
        print("Forwarded contributions.")
    
    def receive_contribution(self, contributor_id: int, contribution: str):
        self.fed_client.receive_contribution(contributor_id, contribution)
    
    # Invoked by the server at the beginning of the initialization phase.
    def initialize(self, public_parameters: dict):
        print("\nInitializing the client...")
        print("Received public parameters:", public_parameters)
        self.fed_client.set_public_parameters(public_parameters)
        # Initialize here.
        print("Initialized.")
    
    def client_update(self, global_model: list):
        # Minimize Err(global_model)
        updated_model = global_model
        return (updated_model, self.client_info.client_id)