import grpc
import numpy

import fedclient
from communication import communication_pb2, helpers
from communication import communication_pb2_grpc

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
        self.client_info = self.server_stub.RegisterClient(
            communication_pb2.ClientAddress(client_address=self.self_addr))
        print("Registered with id", self.client_info.client_id)
        self.fed_client.set_id(self.client_info.client_id)
        # Receive client list.
        for client_info in self.server_stub.ReceiveClients(communication_pb2.VoidMsg()):
            if client_info.client_id == self.client_info.client_id:
                continue
            self.clients[client_info.client_id] = client_info.client_address
        print("Received client list of size", len(self.clients))

    # Called when a new client is joined to the system.
    def new_client(self, new_client_id: int, new_client_addr: str):
        print("Client: New client", new_client_addr, "joined with id", new_client_id)
        # Save the client.
        self.clients[new_client_id] = new_client_addr

    # Invoked by the server at the beginning of the setup phase.
    def setup(self):
        pub_keys = self.fed_client.setup(self.clients)
        self.server_stub.ForwardContributions(
            helpers.serialize_contributions(pub_keys, self.client_info.client_id))

    def receive_contribution(self, contributor_id: int, contribution: str):
        self.fed_client.receive_contribution(contributor_id, contribution)

    # Invoked by the server at the beginning of the initialization phase.
    def initialize(self, pub_params: dict):
        print("\nInitializing the client...")
        print("Received public parameters:", pub_params)
        self.fed_client.set_public_parameters(pub_params)
        self.fed_client.load_data()
        self.fed_client.load_learner()

    def initialize_model(self, model):
        self.fed_client.set_initial_model(model)

    def client_update(self):
        updated_model, weight = self.fed_client.calculate_update()
        return updated_model, weight

    def client_apply_update(self, global_model):
        self.fed_client.apply_update(global_model)
