from pyDH import DiffieHellman
from concurrent import futures

import grpc
from communication import communication_pb2
from communication import communication_pb2_grpc
import public_parameters

server_menu = "\n(s)how the clients\n(i)nitialize\n(se)tup\n(r)un\n(q)uit"

class Server:
    # Maps all the client ids to their addresses.
    clients = {}
    
    def run_interface(self):
        user_input = ""
        while user_input != "q":
            print(server_menu)
            user_input = input("Choose an option:")
            if user_input == "i":
                self.initializate_all()
            elif user_input == "s":
                for client_id, client_addr in self.clients.items():
                    print(str(client_id) + ": " + client_addr)
            elif user_input == "se":
                self.setup_all()
    
    def client_addresses(self):
        return self.clients.values()
    
    def register_client(self, new_addr):
        # Add the new client to the central list of clients.
        new_id = len(self.clients)
        self.clients[new_id] = new_addr
        print("Registering client with address", new_addr, "and with id", new_id)
        # Inform the other clients.
        for clientaddr in filter(lambda x: x != new_addr, self.client_addresses()):
            print("Informing client with address", clientaddr)
            ch = grpc.insecure_channel(clientaddr)
            sb = communication_pb2_grpc.ClientStub(ch)
            sb.NewClient(communication_pb2.ClientInformation(client_id = new_id, client_address = new_addr))
        return new_id
    
    # Returns the public parameters of the system as a dictionary.
    def get_public_parameters(self):
        return {"system_size": len(self.clients), "epoch": 50, "group_desc": 14}

    # Requests all the clients to initialize themselves.
    def initializate_all(self):
        for clientaddr in self.client_addresses():
            ch = grpc.insecure_channel(clientaddr)
            sb = communication_pb2_grpc.ClientStub(ch)
            sb.Initialize(public_parameters.serialize_public_parameters(self.get_public_parameters()))
    
    # Requests all the clients to run the setup phase.    
    def setup_all(self):
        for client_id, client_addr in self.clients.items():
            print("Setting up client", client_id)
            ch = grpc.insecure_channel(client_addr)
            sb = communication_pb2_grpc.ClientStub(ch)
            sb.Setup(communication_pb2.VoidMsg())
        print("Set up done.")
    
    def forward_contribution(self, target_id: int, contributor_id: int, contribution: str):
        target_address = self.clients[target_id]
        ch = grpc.insecure_channel(target_address)
        sb = communication_pb2_grpc.ClientStub(ch)
        sb.ReceiveContribution(communication_pb2.Contribution(target_id = target_id, contributor_id = contributor_id, contribution = contribution))
    
    def execute_round(self):
        import concurrent.futures
        global_weights = [0.0, 0.3, 0.2, 0.8]
        with concurrent.futures.ThreadPoolExecutor(max_workers=len(self.clients)) as exec:
            future = [exec.submit(self.receive_from_client, client_id, global_weights) for client_id in self.clients.keys()]
        # Received all the results from the client!
        for client_result in concurrent.futures.as_completed(future):
            print(client_result.result())

    def receive_from_client(self, client_id, global_weights):
        print("Updating client", client_id, "with weights", global_weights)
        client_addr = self.clients[client_id]
        ch = grpc.insecure_channel(client_addr)
        sb = communication_pb2_grpc.ClientStub(ch)
        updated_model_it = sb.ClientUpdate((communication_pb2.ModelWeight(data_size = -1, value = w) for w in global_weights))
        updated_model = [x for x in updated_model_it]
        updated_weights = [x.value for x in updated_model]
        data_size = updated_model[0].data_size
        print("Received weights", updated_weights, "with data size", data_size)
        return {"client_id": client_id, "updated_weights": updated_weights, "data_size": data_size}