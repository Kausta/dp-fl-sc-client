from pyDH import DiffieHellman
from concurrent import futures

import grpc
from communication import communication_pb2
from communication import communication_pb2_grpc
from communication import rpc_server
import public_parameters

class Server:
    clients = {}
        
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
        return {"group_desc": 14}

    # Requests all the clients to initialize themselves.
    def initializate_all(self):
        for clientaddr in self.client_addresses():
            ch = grpc.insecure_channel(clientaddr)
            sb = communication_pb2_grpc.ClientStub(ch)
            sb.Initialize(public_parameters.serialize_public_parameters(self.get_public_parameters()))
    
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
    
def serve():
    import sys
    server = Server()
    server_port = sys.argv[1] if len(sys.argv) > 1 else "8000"
    print("Listening at", server_port)
    s = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    communication_pb2_grpc.add_ServerServicer_to_server(rpc_server.RpcServer(server), s)
    s.add_insecure_port("[::]:" + server_port)
    s.start()
    while True:
        print("1. Show client list\n2. Initialize\n3. Execute one round")
        userinput = int(input("Choose:"))
        if userinput == 1:
            print("Clients:")
            for i, addr in server.clients.items():
                print("\t", i, "=>", addr)
        elif userinput == 2:
            print("Initializing...")
            server.initializate_all()
        elif userinput == 3:
            server.execute_round()
    s.wait_for_termination()

if __name__ == "__main__":
    serve()