import grpc
from communication import communication_pb2, helpers
from communication import communication_pb2_grpc
import fedserver

server_menu = "\n(s)how the clients\n(i)nitialize\n(se)tup\n(r)un\n(q)uit"


class Server:
    # Maps all the client ids to their addresses.
    clients = {}
    fed_server = fedserver.FedServer()

    def run_interface(self):
        user_input = ""
        while user_input != "q":
            print(server_menu)
            user_input = input("Choose an option:")
            if user_input == "i":
                self.initialize_all()
            elif user_input == "s":
                for client_id, client_addr in self.clients.items():
                    print(str(client_id) + ": " + client_addr)
            elif user_input == "se":
                self.setup_all()
            elif user_input == "r":
                self.execute_round()

    def client_addresses(self):
        return self.clients.values()

    def register_client(self, new_addr):
        # Add the new client to the central list of clients.
        new_id = len(self.clients)
        self.clients[new_id] = new_addr
        print("Registering client with address", new_addr, "and with id", new_id)
        # Inform the other clients.
        for client_addr in filter(lambda x: x != new_addr, self.client_addresses()):
            print("Informing client with address", client_addr)
            ch = grpc.insecure_channel(client_addr)
            sb = communication_pb2_grpc.ClientStub(ch)
            sb.NewClient(communication_pb2.ClientInformation(client_id=new_id, client_address=new_addr))
        return new_id

    # Requests all the clients to initialize themselves.
    def initialize_all(self):
        self.fed_server.set_clients(self.clients)
        for client_addr in self.client_addresses():
            ch = grpc.insecure_channel(client_addr)
            sb = communication_pb2_grpc.ClientStub(ch)
            # Send the public parameters and the initial model.
            public_params = self.fed_server.get_public_parameters()
            sb.Initialize(helpers.serialize_public_parameters(public_params))
            init_model = self.fed_server.get_init_model_flattened()
            sb.InitializeModel(helpers.serialize_model(init_model, -1))

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
        sb.ReceiveContribution(communication_pb2.Contribution(target_id=target_id, contributor_id=contributor_id,
                                                              contribution=contribution))

    def execute_round(self):
        import concurrent.futures
        # Get the updates from the client.
        future = None
        with concurrent.futures.ThreadPoolExecutor(max_workers=len(self.clients)) as executor:
            future = [executor.submit(self.get_client_update, client_id) for client_id in self.clients.keys()]
        # Received all the results from the client!
        updated_models = []
        weights = []
        # Collect all of the updates.
        for client_result in concurrent.futures.as_completed(future):
            updated_model, weight = client_result.result()
            updated_models.append(updated_model)
            weights.append(weight)
        # Aggregate.
        global_update = self.fed_server.aggregate(updated_models, weights)
        # Update all.
        with concurrent.futures.ThreadPoolExecutor(max_workers=len(self.clients)) as executor:
            future = [executor.submit(self.apply_client_update, global_update, client_id) for client_id in self.clients.keys()]
        for client_result in concurrent.futures.as_completed(future):
            client_result.result()
            print("Server: Updated client.")

    def apply_client_update(self, global_update, client_id):
        print("Server: Updating client", client_id)
        client_addr = self.clients[client_id]
        ch = grpc.insecure_channel(client_addr)
        sb = communication_pb2_grpc.ClientStub(ch)
        sb.ClientApplyUpdate(helpers.serialize_model(global_update, -1))

    def get_client_update(self, client_id):
        print("Server: Getting updates from client", client_id)
        client_addr = self.clients[client_id]
        ch = grpc.insecure_channel(client_addr)
        sb = communication_pb2_grpc.ClientStub(ch)
        updated_model, weight = helpers.parse_model(sb.ClientUpdate(communication_pb2.VoidMsg()))
        return updated_model, weight
