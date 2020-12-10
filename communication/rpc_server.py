from communication import communication_pb2
from communication import communication_pb2_grpc
from server import Server

class RpcServer(communication_pb2_grpc.ServerServicer):
    server = None

    def __init__(self, server: Server):
        self.server = server

    def RegisterClient(self, request, ctx):
        cid = self.server.register_client(request.client_address)
        return communication_pb2.ClientInformation(client_address=request.client_address, client_id=cid)

    def ReceivePublicParameters(self, request, ctx):
        import public_parameters
        params = public_parameters.serialize_public_parameters(self.server.public_parameters())
        for p in params:
            yield p

    def ReceiveClients(self, request, ctx):
        for (i, clientaddr) in enumerate(self.server.client_addresses()):
            yield communication_pb2.ClientInformation(client_id=i, client_address=clientaddr)