from communication import communication_pb2
from communication import communication_pb2_grpc
from client import Client
import public_parameters

class RpcClient(communication_pb2_grpc.ClientServicer):
    client = None

    def __init__(self, client: Client):
        self.client = client
        
    def NewClient(self, request, ctx):
        self.client.new_client(request.client_id, request.client_address)
        return communication_pb2.Ack(result = 1)
    
    def Initialize(self, request, ctx):
        self.client.initialize(public_parameters.parse_public_parameters(request))
        return communication_pb2.Ack(result=1)
    
    def ClientUpdate(self, request, ctx):
        updated_model, size = self.client.client_update([x.value for x in request])
        for w in updated_model:
            yield communication_pb2.ModelWeight(value = w, data_size = size)
